import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from pathlib import Path
import keras
from keras.callbacks import ModelCheckpoint
# Internal tools
from util import print, check_gpu_compute, setup_folders
from dataset_instance import DatasetInstance
from tensorflow_func import *

print("TRAINER.PY RUNNING ---------------------------------------------")
setup_folders()
check_gpu_compute()

# SETTINGS ---------------------------------------------

# Path to the data directory
data_dir = Path("./datasets/dataset-N50000-512-64-1-20-8-8-24-28-0-20-F1-arial.ttf")
# Batch size for training and validation
batch_size = 16
initial_epoch = 0
# Desired image dimensions
img_width = 64
img_height = 8
train_data_amount = 0.9

# Create model object based on dataset and settings
dataset_model = DatasetInstance(data_dir, batch_size, img_width, img_height, train_data_amount)
dataset_model.visualize_training_dataset()

# Get the model of the neural network
model = build_model(img_width, img_height, dataset_model.char_to_num)
model.summary()


# Training ---------------------------------------------

# TODO: restore epoch count.
epochs = 100
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)
# Reduce learning rate
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    patience=5,
    factor=0.5,
    min_lr=1e-6,
    verbose=1
)

# Model Checkpoint
checkpoint_path = "model_checkpoint.weights.h5"
model_checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='auto',
    save_freq='epoch',
)

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.input["image"], model.get_layer(name="dense_output").output
)

# Initialize the custom callback
visualize_predictions = PredictionVisualizer(
    model=prediction_model,
    validation_data=dataset_model.validation_dataset,
    max_length=dataset_model.max_length,
    num_to_char=dataset_model.num_to_char
)

callbacks = [model_checkpoint, visualize_predictions, reduce_lr]

if os.path.exists("model_checkpoint.weights.h5"):
    print("Weights found, loading...")
    model.load_weights('model_checkpoint.weights.h5')
    # TODO: add initial_epoch to filename, dynamic loading
else:
    print("No weights found, creating new weights...")
    initial_epoch = 0

print("Starting training...")

# Train the model
history = model.fit(
    dataset_model.train_dataset,
    validation_data=dataset_model.validation_dataset,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    initial_epoch=initial_epoch  # Resume from here
)

model.save_weights('model.weights.h5')

print("Exited...")
