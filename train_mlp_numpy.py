################################################################################
# MIT License
#
# Copyright (c) 2025 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2025
# Date Created: 2025-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import matplotlib.pyplot as plt
import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    pred_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(pred_labels == targets)
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    total_correct = 0
    total_samples = 0
    for Xb, yb in data_loader:
        # Xb: [B, C, H, W] numpy; flatten:
        Xb = Xb.reshape(Xb.shape[0], -1).astype(np.float32)
        logits = model.forward(Xb)   
        preds = np.argmax(logits, axis=1)
        total_correct += np.sum(preds == yb)
        total_samples += Xb.shape[0]
        
    avg_accuracy = total_correct / max(total_samples, 1)
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_dict: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    model = MLP(n_inputs=32*32*3, n_hidden=hidden_dims, n_classes=10)
    loss_module = CrossEntropyModule()
    # TODO: Training loop including validation
    val_accuracies = []
    val_losses = []
    train_losses = []
    train_accuracies = []
    best_state = []
    best_val = -1.0
    
    def sgd_step(model, lr):
        for m in getattr(model, 'modules', []):
            if hasattr(m, 'params') and hasattr(m, 'grads'):
                if m.params.get('weight') is not None:
                    m.params['weight'] -= lr * m.grads['weight']
                if m.params.get('bias') is not None:
                    m.params['bias'] -= lr * m.grads['bias']
    
    
    for epoch in range(epochs):
        epoch_losses = []
        correct_train = 0
        total_train = 0

        for Xb, yb in cifar10_loader['train']:
            B = Xb.shape[0]
            Xb = Xb.reshape(B, -1).astype(np.float32)

            probs = model.forward(Xb)
            loss = loss_module.forward(probs, yb)
            epoch_losses.append(loss)

            preds = np.argmax(probs, axis=1)
            correct_train += np.sum(preds == yb)
            total_train += B

            dprobs = loss_module.backward(probs, yb)
            _ = model.backward(dprobs)
            sgd_step(model, lr)

        # store per-epoch train stats
        train_losses.append(float(np.mean(epoch_losses)))
        train_accuracies.append(correct_train / total_train)

        # ---- Validation ----
        val_acc = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(float(val_acc))

        # compute validation loss (optional but small overhead)
        val_epoch_losses = []
        for Xv, yv in cifar10_loader['validation']:
            Bv = Xv.shape[0]
            Xv = Xv.reshape(Bv, -1).astype(np.float32)
            pv = model.forward(Xv)
            val_epoch_losses.append(loss_module.forward(pv, yv))
        val_losses.append(float(np.mean(val_epoch_losses)))

        
      
        # keep best
        if val_acc > best_val:
            best_val = val_acc
            # snapshot params
            best_state = []
            for m in model.modules:
                if hasattr(m, 'params'):
                    # deep copy numeric arrays
                    ps = {}
                    for k, v in m.params.items():
                        ps[k] = None if v is None else v.copy()
                    best_state.append(ps)
                else:
                    best_state.append(None)
    
    # restore best
    idx = 0
    for m in model.modules:
        if hasattr(m, 'params') and best_state[idx] is not None:
            for k in m.params.keys():
                m.params[k] = best_state[idx][k]
        idx += 1
    
    # TODO: Test best model
    test_accuracy = evaluate_model(model, cifar10_loader['test'])
    
    # TODO: Add any information you might want to save for plotting
    
    # Logging (for plotting)
    logging_dict = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': float(best_val),
        'test_accuracy': float(test_accuracy),
    }

    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    _, _, _, logging_dict = train(**kwargs)
    
    os.makedirs(name="assets/plots", exist_ok=True)
    
    def plot_training_curves(logging_dict, save_path='assets/plots/training_curves.png'):
        """
        Plots training/validation losses and accuracies on a single plot.
        """
        epochs = range(1, len(logging_dict["train_losses"]) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, logging_dict["train_losses"], 'b-o', label="Train Loss")
        plt.plot(epochs, logging_dict["val_losses"], 'c--o', label="Val Loss")
        plt.plot(epochs, logging_dict["train_accuracies"], 'r-s', label="Train Acc")
        plt.plot(epochs, logging_dict["val_accuracies"], 'm--s', label="Val Acc")

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training and Validation Curves")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved training curve to {save_path}")

    
    plot_training_curves(logging_dict=logging_dict)