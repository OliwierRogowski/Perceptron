# PERCEPTRON

[![C++](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)
[![Eigen](https://img.shields.io/badge/Library-Eigen-orange.svg)](https://eigen.tuxfamily.org/)
[![Philosophy](https://img.shields.io/badge/Philosophy-No%20Black%20Boxes-red.svg)](#)

A mathematical implementation of a **Multi-Layer Perceptron (MLP)** built from absolute scratch. This project is a part of my "No Black Boxes" journey, where I implement core AI algorithms using only **C++** and **Linear Algebra** to fully grasp the underlying mechanics of machine learning.

## 🚀 Key Features
- **Pure C++ Implementation:** No reliance on heavy frameworks like PyTorch or TensorFlow.
- **Matrix Operations via Eigen:** High-performance linear algebra for weight updates and forward passes.
- **Manual Backpropagation:** Custom implementation of the training loop and gradient descent.
- **Dynamic Architecture:** Ability to define any network topology (input, hidden layers, output).
- **MNIST-Ready:** Designed to handle complex classification tasks like handwritten digit recognition.

## 🏗️ Technical Deep Dive
The engine implements a standard feed-forward architecture with a Sigmoid activation function. The training process uses the **Chain Rule** to propagate errors back through the network:

```cpp
// Manual Gradient Calculation in my code:
Eigen::VectorXd delta = (target - output).array() * output.array() * (1.0 - output.array());
// ... followed by weight and bias updates
