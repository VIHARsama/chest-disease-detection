# chest-disease-detection

A fully functional project to detect chest diseases using neural networks. The model is trained on a dataset of chest X-rays to classify images as either normal or abnormal. This classification can help in the early detection of pneumonia, tuberculosis, covid and lung cancer.

## Table of Contents

- [Project Description](#project-description)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)

## Project Description

Chest disease detection is a critical task in the medical field. The project uses a neural network to classify chest X-ray images and identify whether the image represents a healthy lung or one with a disease. This approach can assist radiologists and healthcare professionals in diagnosing lung-related issues more efficiently.

The neural network is trained on labeled chest X-ray images, with the model learning to extract patterns and features that distinguish healthy lungs from those with diseases.

## Technologies Used

- **Python** (for implementation)
- **TensorFlow / Keras** (for building and training the neural network)
- **OpenCV** (for image processing)
- **NumPy** (for numerical computations)
- **Matplotlib** (for visualizations)
- **scikit-learn** (for model evaluation)

## Dataset

This project uses a chest X-ray dataset, which includes images of healthy lungs and those with various diseases. A commonly used dataset for this purpose is the [Chest X-ray Images (Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle, which contains thousands of images labeled as `normal` or `pneumonia`.

You can download the dataset from the link above or use any similar dataset.

## Model Architecture

The model is based on a Convolutional Neural Network (CNN) that performs well in image classification tasks. It consists of:

1. **Convolutional layers** for feature extraction.
2. **Pooling layers** to reduce spatial dimensions and retain important information.
3. **Fully connected layers** for decision-making and classification.

The model architecture is designed to detect patterns in chest X-ray images, classify the presence or absence of chest diseases, and provide high accuracy.

## Result

| Disease      | Accuracy | Recall | Precision | F1  |
| ------------ | -------- | ------ | --------- | --- |
| Covid        | 96%      | 96%    | 95%       | 99% |
| Tuberculosis | 96%      | 92%    | 94%       | 92% |
| Pneumonia    | 97%      | 98%    | 98%       | 98% |
| Cancer       | 99%      | 98%    | 99%       | 98% |

## Usage

1. Clone the repository to your local machine:
```
git clone https://github.com/VIHARsama/chest-disease-detection.git
```

2. Navigate to the project directory:

```
cd chest-disease-detection
```

3. Install the following dependencies
	- Python
	- Flask
	- Tensorflow
	- Numpy
	- Werkzeug
	- OpenCV

4. Execute the following command in your terminal:

```
python app.py
```
        
5. Open your browser and go to `http://127.0.0.1:5000` to use the app locally.
