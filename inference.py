import math
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import keras
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from rich.console import Console
from rich.theme import Theme
# Internal tools
from util import print
from dataset_instance import DatasetInstance
from tensorflow_func import *
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from pathlib import Path


charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
batch_size = 1
train_data_amount = 0

def get_folder_path():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(
        title="Select folder with pixelated images"
    )
    return folder_path

folder_path = get_folder_path()
if not folder_path:
    print("No folder selected.")
    exit()

folder_path = Path(folder_path)  # Convert to Path object

# Proceed as before
image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
image_paths = [p for p in folder_path.iterdir() if p.suffix.lower() in image_extensions]
if not image_paths:
    print("No images found in the selected folder.")
    exit()

# Load the first image and get its dimensions
with Image.open(image_paths[0]) as img:
    img_width, img_height = img.size

# Create model object based on dataset and settings
dataset_model = DatasetInstance(folder_path, batch_size, img_width, img_height, train_data_amount, charset)

# Get the model of the neural network
model = build_model(img_width, img_height, dataset_model.char_to_num)
model.summary()

# Load model ---------------------------------------------

if os.path.exists("model_checkpoint.weights.h5"):
    model.load_weights('model_checkpoint.weights.h5')
    print("MODEL LOADED!")
else:
    print("NO MODEL FOUND!")

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.input["image"], model.get_layer(name="dense_output").output
)

batches = len(dataset_model.validation_dataset)
validation_images = batches * batch_size

custom_theme = Theme({
    "green": "green",
    "light_green": "green1",
    "white": "bold white"
})
console = Console(theme=custom_theme)
progressbar = Progress(
    TextColumn("[progress.description]{task.description}", style="white"),
    BarColumn(bar_width=None, complete_style="green", finished_style="light_green"),
    TaskProgressColumn(style="green"),
    TextColumn("[bold white]Elapsed: "),
    TimeElapsedColumn(),
    TextColumn("[bold white]ETA: "),
    TimeRemainingColumn()
)

output_dir = Path('inference_results')
output_dir.mkdir(exist_ok=True)

with progressbar as progress:
    task = progress.add_task(f"Processing 0 / {validation_images}", total=validation_images)
    for batch_num, batch in enumerate(dataset_model.validation_dataset):
        batch_images = batch["image"]
        batch_labels = batch["label"]

        preds = prediction_model.predict(batch_images, verbose=0)
        pred_texts = decode_batch_predictions(preds, dataset_model.max_length, dataset_model.num_to_char)
        orig_texts = []
        for label in batch_labels:
            label = tf.strings.reduce_join(dataset_model.num_to_char(label)).numpy().decode("utf-8")
            orig_texts.append(label)

        for i in range(len(batch_labels)):
            pred_text = pred_texts[i].replace("[UNK]", "")
            img = (batch_images[i] * 255).numpy().astype(np.uint8)

            # Set the canvas size
            fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
            ax.imshow(img[:, :, 0], cmap="gray")
            ax.set_title(pred_text, fontsize=14, color="black", loc='center', pad=5)
            ax.axis("off")

            save_path = output_dir / f'{orig_texts[i]}.png'
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        progress.update(task, advance=batch_size, description=f"Processing {(batch_num + 1) * batch_size} / {validation_images}")

print("Inference completed.")

# Displaying the images
image_paths = sorted(output_dir.glob('*.png'))
num_images = len(image_paths)

if num_images == 0:
    print("No images found in the output directory.")
else:
    # Define the number of columns for the grid
    cols = 5  # Adjust as needed
    rows = math.ceil(num_images / cols)

    # Set the figure size
    plt.figure(figsize=(cols * 6, rows * 6))

    for idx, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        plt.subplot(rows, cols, idx + 1)
        if img.ndim == 2:  # Grayscale
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.axis('off')

    plt.tight_layout()
    plt.show()