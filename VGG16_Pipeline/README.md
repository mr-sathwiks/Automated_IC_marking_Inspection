#IC Chip Image Rotation Correction using VGG16

This project implements an automated rotation correction module for integrated circuit (IC) chip images using a deep learning approach based on the VGG16 convolutional neural network. The model classifies images into one of four major orientations (0°, 90°, 180°, 270°) and rotates them to a canonical upright position, enabling robust downstream processing for tasks such as marking inspection and classification.

Features
Deep Learning-Based Orientation Detection:
Utilizes a VGG16 architecture, fine-tuned for four-class orientation classification (0°, 90°, 180°, 270°).

Transfer Learning:
Leverages pretrained weights from ImageNet for faster convergence and improved accuracy.

Early Stopping:
Custom early stopping function (stopIfAccuracyNotImproving) halts training when validation accuracy plateaus.

Data Augmentation:
Includes random translations, scaling, and X reflection to improve robustness.

High Accuracy:
Achieves over 99% accuracy on held-out test data.

How It Works
Data Preparation:

Images are labeled according to their orientation (0, 90, 180, 270 degrees).

Data is split into training and validation sets, with augmentation applied to the training set.

Model Architecture:

VGG16 backbone with the final layer modified for 4-class output.

Softmax activation for multi-class classification.

Training:

Uses stochastic gradient descent with momentum (SGDM).

Employs L2 regularization and learning rate scheduling.

Early stopping based on validation accuracy stability.

Inference:

For each input image, the model predicts the orientation class.

The image is rotated to the upright (0°) orientation using the predicted class.
