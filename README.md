# Depixelation Tool

## Description
A machine learning tool designed to recognize text from heavily pixelated images. This tool generates datasets of pixelated text images, trains a neural network model to recognize the text, and provides benchmarking capabilities to evaluate the model's performance.

The model is built using TensorFlow and Keras, leveraging convolutional neural networks (CNNs) and recurrent neural networks (RNNs) with connectionist temporal classification (CTC) loss for sequence prediction.

You can customize all parameters directly in the files.

## Usage
Files with `launch_` prefix will install dependencies for the whole project automatically.

If you are not interested in training your own model, then you can use an existing model and run `launch_inference.py`.

### Dataset Generation
Generate a dataset of pixelated text images using `data_generator.py`.

### Training the Model
Train the OCR model using `launch_trainer.py`

### Benchmarking
Evaluate the trained model using `launch_benchmarker.py`

### Inference
Depixelate a single image using `launch_inference.py`
