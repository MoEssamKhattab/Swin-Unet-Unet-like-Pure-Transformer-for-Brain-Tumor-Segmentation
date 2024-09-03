# Swin-Unet: A Pure Transformer-Based Model for Brain Tumor Segmentation

This repository contains the implementation of **Swin-Unet**, a novel model designed for brain tumor segmentation using 2D medical images. Swin-Unet leverages the power of the Swin Transformer within a U-Net-like architecture, providing significant improvements in accuracy and generalization over traditional convolutional neural networks (CNNs).

## Overview

### Abstract
Recent advancements in the Computer Vision field, particularly in medical image segmentation, have predominantly relied on CNNs. Despite their success, CNNs often face challenges in capturing long-range dependencies due to the locality of convolution operations. Swin-Unet addresses these limitations by introducing a pure Transformer-based approach, where the Swin Transformer is utilized in a U-shaped architecture. This model effectively processes input images by dividing them into non-overlapping patches, each treated as a token, and then applies Transformer-based encoders and decoders to produce highly accurate segmentation maps. Swin-Unet has demonstrated a 98.9% accuracy in brain tumor segmentation tasks, outperforming existing CNN-based methods.

### Key Features
- **Transformer-Based U-Net Architecture**: Utilizes the Swin Transformer in an encoder-decoder setup with skip connections.
- **High Accuracy**: Achieves 98.9% accuracy on brain tumor segmentation datasets.
- **Efficient Computation**: Incorporates the Swin Transformer's shifted window mechanism for improved performance.

## Dataset

The Swin-Unet model is trained and validated on the LGG Segmentation Dataset available on Kaggle. This dataset contains preoperative MRI scans of brain tumors and can be accessed <a href="https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation">here</a>.

### Dataset Details
* Source: The dataset includes data from The Cancer Genome Atlas (TCGA) and The Cancer Imaging Archive (TCIA).
* Imaging Modalities: Multiple modalities are included, with FLAIR as the primary sequence when others are missing.
* Annotations: Tumor regions are manually annotated and verified by radiologists.

## Model Architecture

The Swin-Unet architecture is a combination of the following components:

* Encoder: Built using Swin Transformer blocks, it processes image patches to extract deep features.
* Bottleneck: Connects the encoder and decoder, applying additional Transformer blocks.
* Decoder: Upsamples the encoded features using a novel patch-expanding layer to restore image resolution.
* Skip Connections: Transfers multi-scale features from the encoder to the decoder to maintain spatial information.

## Training Process

### Data Preprocessing

* Format Conversion: Images and masks are converted to PNG format.
Image Resizing: All images are resized to 224x224 pixels to standardize input dimensions.
* Grayscale Conversion: Masks are converted to grayscale to reduce model complexity.
* Data Augmentation: Techniques such as rotation, scaling, and horizontal flipping are applied to improve model robustness.

### Training Details
* Optimizer: The model is trained using the Adam optimizer with a learning rate of 0.001.
* Loss Function: Binary cross-entropy is used as the loss function, ideal for binary segmentation tasks.
* Hardware: Training was conducted on NVIDIA GPU P100 accelerators provided by Kaggle.

## Results
The Swin-Unet model achieved a validation accuracy of 98.9% on the brain tumor segmentation task, surpassing other architectures. Below is a comparison of accuracy metrics:

<table>
    <tr>
        <th>Model</th>
        <th>Accuracy</th>
    </tr>
    <tr>
        <td>ViT-based DNN</td>
        <td>97.98%</td>
    </tr>
    <tr>
        <td>QFS-Net</td>
        <td>98.23%</td>
    </tr>
    <tr>
        <td>Swin-Unet</td>
        <td>98.9%</td>
    </tr>
</table>

## Conclusion
Swin-Unet presents a significant advancement in medical image segmentation by successfully integrating Transformer-based models into this domain. Its high accuracy and robust performance demonstrate its potential to aid in the early detection and diagnosis of brain tumors, contributing to improved patient outcomes.