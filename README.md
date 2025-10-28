# MLPs and Backpropagation

This repository focuses on building, training, and analyzing Multi-Layer Perceptrons (MLPs) both from scratch using NumPy and using PyTorch.

---

## 📘 Overview

This repository introduces the theoretical and practical foundations of backpropagation and optimization in neural networks.

- Derive the mathematical equations of backpropagation from first principles

- Implement a full MLP classifier in NumPy, including manual gradient computation

- Re-implement the same architecture in PyTorch using automatic differentiation

- Analyze optimization dynamics, saddle points, and batch normalization

- Evaluate and compare performance across implementations

---

## 🧩 Project Structure

```graphql
├── modules.py               # Core MLP components (Linear, Activation, Softmax)
├── mlp_numpy.py             # NumPy-based MLP implementation
├── train_mlp_numpy.py       # Training/testing for NumPy MLP
├── mlp_pytorch.py           # PyTorch MLP implementation
├── train_mlp_pytorch.py     # Training/testing for PyTorch MLP
├── unittests.py             # Provided tests for correctness
├── env.yaml                 # Conda environment with dependencies
└── a1.pdf                   # Assignment instructions
```

---

## ⚙️ Setup

#### 1. Create the environment
if using cpu
```shell
conda env create -f dl2025_cpu.yml
conda activate mlp
```

if using cuda: 

```shell
conda env create -f dl2025_gpu.yml
conda activate mlp
```

#### 2. Install additional packages (if needed)
```shell
pip install torch matplotlib numpy
```

#### 3. 🚀 Running the Code

- NumPy implementation:

    ```shell
    python train_mlp_numpy.py
    ```

- PyTorch implementation:
    ```shell
    python train_mlp_pytorch.py
    ```

Both scripts train a simple MLP with:

- 1 hidden layer of 128 units

- ELU activations

- Softmax output

- Cross-entropy loss

- Learning rate: 0.1

- 10 epochs

Expected test accuracy ≈ 47–48% on the provided dataset (ans-delft).

---

## 📊 Experiments

- NumPy MLP: Manual gradient implementation and visualization of loss curves.

- PyTorch MLP: GPU-accelerated training and comparison of learning dynamics.

- Batch Normalization: Evaluated its impact on convergence and neuron activity.

- Saddle Points: Discussed optimization challenges and eigenvalue effects.

---

## 🧠 Learning Outcomes

- Understanding of gradient derivations and index notation

- Implementation of backpropagation without autodiff

- Comparison of analytical vs. automatic gradient computation

- Insights into optimization behavior and normalization techniques

---

## 🧾 References

- Matrix Cookbook — [Wolkowicz et al.](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

- Mathematics for Machine Learning — [Deisenroth et al.](#)

- [PyTorch Documentation](https://pytorch.org/) 

- [NumPy einsum Guide](https://ajcr.net/Basic-guide-to-einsum/)