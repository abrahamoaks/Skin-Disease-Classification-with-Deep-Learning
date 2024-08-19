# Skin Disease Classification with Deep Learning -- Abraham Obianke 

## Introduction

In this project, I aimed to classify skin diseases using various deep learning models. The goal was to develop a model that can accurately categorize images of skin conditions into nine predefined classes. This involved data preparation, model training, evaluation, and analysis of results.

## Dataset

The dataset used is the ISIC 2019 dataset, which includes a diverse collection of images depicting different skin diseases. I split the dataset into training and validation sets to train and evaluate the models effectively. Each image was resized to 240x240 pixels and normalized for consistent input to the models.
*Data Source: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data*

## Procedures and Approaches

### Data Preparation

1. **Data Reduction and Splitting:** I reduced the dataset size to manage computational resources and split it into training and validation subsets.
2. **Image Processing:** Images were resized to 240x240 pixels, converted to RGB, and normalized.

### Model Architectures

1. **DenseNet121:**
   - **Architecture:** Utilized DenseNet121 with pre-trained weights from ImageNet as the base model. Added Global Average Pooling, Dropout for regularization, and a Dense layer with softmax activation for classification.
   - **Training:** The model was compiled with categorical crossentropy loss and Adam optimizer. It was trained for 5 epochs with a validation split to monitor performance.

2. **MobileNet:**
   - **Architecture:** Used MobileNet as the base model, with additional Dropout, Convolution, and Global Average Pooling layers. The final Dense layer used softmax activation.
   - **Training:** The MobileNet model was also compiled with categorical crossentropy and trained for 5 epochs.

3. **Custom CNN:**
   - **Architecture:** Designed a custom Convolutional Neural Network (CNN) with multiple convolutional layers, max pooling, and dropout for regularization. Ended with Flattening and Dense layers for classification.
   - **Training:** The custom CNN was compiled with categorical crossentropy and trained similarly for 5 epochs.

### Model Evaluation

I evaluated each model on the validation dataset to assess its performance. Metrics such as accuracy and loss were recorded to compare the effectiveness of each architecture.

### Prediction and Analysis

1. **Confusion Matrix:** I generated confusion matrices for each model to analyze prediction performance and identify areas where models might be misclassifying.
2. **Observations:** I observed varying performance across different models. DenseNet121 showed promising results with higher accuracy, whereas the custom CNN had lower performance. 

## Usefulness

This classification project is useful for automating the diagnosis of skin diseases, potentially aiding dermatologists in early detection and treatment planning. The models developed can be integrated into diagnostic tools to enhance medical practice efficiency.

## Conclusion

Through this project, I successfully built and evaluated three deep learning models for skin disease classification. DenseNet121 outperformed the other models, demonstrating its effectiveness in handling complex image classification tasks.
