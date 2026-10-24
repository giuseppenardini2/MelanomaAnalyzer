# MelanomaAnalyzer

## Table of contents
- [About](#about)
- [Dataset](#dataset)
- [License](#license)

## About
Melanoma Analyzer enables the binary classification of moles into melanomas and non-melanomas using a CNN.
The proposed system consists of two interconnected
macro-components:
* Backend (CNN Model): The core of the system is a
custom Convolutional Neural Network (CNN), designed
and implemented from scratch using the
PyTorch framework. The model was trained on a
balanced dataset containing over 10,000 images, utilizing
data augmentation techniques (rotations, flips,
color variations) to maximize its
generalization ability and robustness against variations
in image acquisition.
* Frontend (Web App): A web interface based on the
Flask micro-framework allows the end user to upload an image
of a skin lesion and receive the result provided 
by the neural network.

> [!WARNING]  
> MyFinger is a university project focused on creating a CNN.
> It is not for medical purpose.

## Dataset
The public [Dermel](#https://www.kaggle.com/datasets/drscarlat/melanoma) dataset was
used for model training and validation.

## License
*Copyright© 2026 Giuseppe Nardini*\
This program is licensed under the [AGPL-3.0](LICENSE) license.
