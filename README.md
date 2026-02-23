Flower Classification Web Application
A Flask-based web application for classifying flower images using a fine-tuned EfficientNet-B0 model trained on the Oxford Flowers-102 dataset.

Overview
This project implements a deep learning-powered flower classification system that can identify 102 different flower species. The application features a user-friendly web interface where users can upload flower images and receive predictions with detailed information about the identified species.

Features
Accurate Classification: Uses EfficientNet-B0 architecture fine-tuned on the Oxford Flowers-102 dataset
Top-K Predictions: Displays the top 5 most likely flower species with confidence scores
Rich Information: Provides detailed descriptions of identified flowers, fetched from Wikipedia when available
User-Friendly Interface: Clean, responsive web interface built with HTML, CSS, and Flask
Image Preview: Shows the uploaded image alongside prediction results
Flexible Input: Supports both file upload and base64-encoded image data


The model:

Uses EfficientNet-B0 as the backbone (pre-trained on ImageNet)
Adds custom classification layers with batch normalization and dropout
Processes 224×224 pixel images
Outputs predictions for 102 flower categories

Installation
Prerequisites
Python 3.8 or higher
CUDA-capable GPU (optional, for faster inference)

Setup
Clone the repository:
Create a virtual environment:
Install dependencies:
Download the dataset (optional, for training):

The Oxford Flowers-102 dataset should be placed in the dataset directory
Organized into train/, valid/, and test/ subdirectories
Ensure model weights exist:

The trained model should be at efficientnetb0_flowers_final.pth
If not available, train the model using flower_efficientnet.ipynb


