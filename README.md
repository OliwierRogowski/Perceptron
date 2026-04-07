# Perceptron Classifier (C++)
## Overview
This project is an implementation of Rosenblatt's Perceptron, a fundamental supervised learning algorithm used for binary classification. The Implementation utilizes the Eigen library for efficient linear algebra operations.
The modle is specifically designed to demonstrate the learning process on linearly separable datasetsm such as the famous Iris Flower Dataset.

## Core Logic and Features
The implementation follows the classi artiffical neuron model:
  1. Weight initialization: Weights are initialized with small and ranodm values using normal distribiution with seed to prevent symmetry and get same result every time when user turn on model
  2. Learing rule: The model updates weights based on error between the predicted label and acual label. It's famous formula: $$\Delta w = \eta \cdot (target - predicted) \cdot x$$
  3. Activation function: function that map[s input to class labels -1 or 1

Implementation details
  1. Bias handling: The bias is integrated into the weight vector as the first element (w_(0), allowing for streamlined matrix operations.
  2. Optimization: Uses Eigen::VectorXd and Eigen::MatrixXd for optimized dot product calculations (net_input)
  3. Convergance Tracking: The model records the number of missclassifications in each epoch, allowing for the analisis of the learning curve.
