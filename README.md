# Depixelation Tool

![LSAB901GQFM7FKDBDMQN](https://github.com/user-attachments/assets/dfb1bb63-da0e-40ba-8db4-44aaa952e792)

## Description
A machine learning tool designed to recognize text from heavily pixelated images. This tool generates datasets of pixelated text images, trains a neural network model to recognize the text, and provides benchmarking capabilities to evaluate the model's performance.

The model is built using TensorFlow and Keras, leveraging convolutional neural networks (CNNs) and recurrent neural networks (RNNs) with connectionist temporal classification (CTC) loss for sequence prediction, inputs are converted to monochrome.

You can customize all parameters directly in the files.

___

## Usage
Files with `launch_` prefix will install dependencies for the whole project automatically.

If you are not interested in training your own model, then you can use an existing model and run `launch_inference.py`.

Different models and image sizes are currently incompatible.
The included model weights are for 64x8px images with alphanumberic text and block average pixelation.
Using a special model designed for a specific characterset is essential for best results.
For example testing on a number input: A general alphanumeric model may read 82% strings correctly, while a special numeric model may yield 96%.

### Dataset Generation
Generate a dataset of pixelated text images using `data_generator.py`.

### Training the Model
Train the OCR model using `launch_trainer.py`

### Benchmarking
Evaluate the trained model using `launch_benchmarker.py`

### Inference
Depixelate a folder of images using `launch_inference.py`
