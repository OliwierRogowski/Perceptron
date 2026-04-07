# Perceptron Classifier (C++)
## Overview
This project is an implementation of Rosenblatt's Perceptron, a fundamental supervised learning algorithm used for binary classification. The Implementation utilizes the Eigen library for efficient linear algebra operations.
The modle is specifically designed to demonstrate the learning process on linearly separable datasetsm such as the famous Iris Flower Dataset.

## Core Logic and Features
The implementation follows the classi artiffical neuron model:
  1. Weight initialization: Weights are initialized with small and ranodm values using normal distribiution with seed to prevent symmetry and get same result every time when user turn on model
  2. Learing rule: The model updates weights based on error between the predicted label and acual label. It's famous formula: $$\Delta w = \eta \cdot (target - predicted) \cdot x$$
