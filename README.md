# Skin Disease Classification with Deep Learning

## Overview

This project contains the implementation of a deep learning model designed to classify various skin diseases from images. The model leverages convolutional neural networks (CNNs) to accurately identify skin conditions, which can aid dermatologists and healthcare professionals in diagnostics. This project includes data preprocessing, model training, evaluation, and deployment scripts, enabling end-to-end skin disease classification.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Dataset

The dataset used in this project consists of skin disease images with corresponding labels. The images are categorized into multiple classes representing different skin conditions. Each image is preprocessed to a consistent size and format to ensure compatibility with the CNN model.

### Dataset Source

The dataset can be obtained from publicly available skin disease image databases such as the [ISIC Archive](https://www.isic-archive.com/)

### Preprocessing

- **Resizing**: All images was resized to a uniform size (e.g., 224x224 pixels) to match the input requirements of the model.
- **Normalization**: Pixel values was normalized to a range of [0, 1] or standardized based on the mean and standard deviation of the dataset.
- **Augmentation**: Techniques such as rotation, flipping, and zooming was applied to increase the diversity of the training set.

## Requirements

The project is implemented in Python using several popular libraries such as:
- Python 3.8+
- TensorFlow 2.x / PyTorch
- NumPy
- Pandas
- scikit-learn
- OpenCV
- Matplotlib

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Installation

1. **Clone the repository:**

    ```bash
    git clone [[https://github.com/yourusername/Skin-Disease-Classification.git](https://github.com/abrahamoaks/Skin-Disease-Classification-with-Deep-Learning)](https://github.com/abrahamoaks/Skin-Disease-Classification-with-Deep-Learning)
    cd Skin-Disease-Classification
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset** (if not already included):

   Place the dataset in the `data/` directory, ensuring it is structured appropriately (e.g., `data/train/class_name/`).

## Model Architecture

The model is built using a convolutional neural network (CNN) architecture. The default model architecture is based on the popular **ResNet50** or **VGG16** network, pre-trained on ImageNet, and fine-tuned on the skin disease dataset.

### Customization

- You can modify the architecture by editing the `model.py` file.
- Transfer learning can be enabled or disabled depending on the dataset size.

## Training

To train the model, run the following command:

```bash
python train.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

### Training Parameters

- **epochs**: Number of epochs for training (default: 50)
- **batch_size**: Batch size for training (default: 32)
- **learning_rate**: Learning rate for the optimizer (default: 0.001)
- **model_checkpoint**: Path to save the best model (default: `checkpoints/model_best.h5`)

### Monitoring

Training progress is logged, and metrics such as accuracy and loss are visualized using TensorBoard or Matplotlib.

## Evaluation

After training, the model can be evaluated on a test set:

```bash
python evaluate.py --model checkpoints/model_best.h5 --test_data data/test/
```

### Metrics

- **Accuracy**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**

The evaluation script will output these metrics and save a confusion matrix plot.

## Results

Results from the model training and evaluation will be saved in the `results/` directory, including:

- Model accuracy and loss plots.
- Confusion matrix.
- Sample predictions.

## Usage

To classify a new image, use the `predict.py` script:

```bash
python predict.py --image_path /path/to/image.jpg --model checkpoints/model_best.h5
```

The script will output the predicted class along with the associated confidence score.

## Deployment
- **Flask Web App**: A friendly user web interface was created using Flask to serve the model for real-time predictions.

- ## Conclusion
In this project, I successfully developed a deep learning model for skin disease classification, achieving reliable results in identifying various conditions from images. By utilizing a CNN architecture with transfer learning, I demonstrated the potential of AI to support dermatologists in diagnostic tasks.

The model's performance was validated through thorough training and evaluation, and I provided deployment scripts for easy integration into healthcare systems. This project not only met its objective but also laid the foundation for future enhancements, showcasing the impactful role of AI in medical diagnostics.
