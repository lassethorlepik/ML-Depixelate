import builtins
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
import tensorflow_func as tf
import keras
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from rich.console import Console
from rich.theme import Theme
# Internal tools
from util import check_gpu_compute, print, setup_folders
from dataset_instance import DatasetInstance
from tensorflow_func import *
matplotlib.use('Agg')

print("BENCHMARKER.PY RUNNING ---------------------------------------------")
setup_folders()
check_gpu_compute()

# Path to the data directory
data_dir = Path("./datasets/dataset-N200000-512-64-64-8-1-20-8-8-24-28-0-20-F1-arial.ttf")
# Batch size for validation, one batch is output as a single image
batch_size = 32
# Image dimensions
img_width = 64
img_height = 8
train_data_amount = 0.97  # Speeds up process as we only us validation here

# Create model object based on dataset and settings
dataset_model = DatasetInstance(data_dir, batch_size, img_width, img_height, train_data_amount)
dataset_model.visualize_training_dataset()

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


# Inference ---------------------------------------------

print(f"RUNNING INFERENCE...")

#  Let's check results on some validation samples
total_correct = 0
total_images = 0
total_match_percentage = 0.0
match_percentages = []  # To store all match percentages
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

        n_rows = (len(orig_texts) + 3) // 4
        _, ax = plt.subplots(n_rows, 4, figsize=(10, n_rows * 1.25))
        for i in range(len(pred_texts)):
            pred_text = pred_texts[i].replace("[UNK]", "")
            orig_text = orig_texts[i].replace("[UNK]", "")
            # Calculate match percentage
            match_percentage = similarity(pred_text, orig_text)
            match_percentages.append(match_percentage)
            total_match_percentage += match_percentage
            img = (batch_images[i] * 255).numpy().astype(np.uint8)
            is_correct = pred_text == orig_text
            color = 'green' if is_correct else 'red'
            title = f"Pred: {pred_text}\nTrue: {orig_text}\nMatch: {match_percentage:.2f}%"
            ax[i // 4, i % 4].imshow(img[:, :, 0], cmap="gray")
            ax[i // 4, i % 4].set_title(title, fontsize=8, color=color)
            ax[i // 4, i % 4].axis("off")

            if is_correct:
                total_correct += 1
            total_images += 1
        
        progress.update(task, advance=batch_size, description=f"Processing {(batch_num + 1) * batch_size} / {validation_images}")
        plt.tight_layout()
        plt.savefig(f'benchmark/batch_{batch_num}.png')
        plt.close()
# After the loop, print a newline character to move the cursor to the beginning of the next line
builtins.print()


# Calculate overall metrics
accuracy_percentage = (total_correct / total_images) * 100
average_match_percentage = total_match_percentage / total_images

# Analyze worst performances
sorted_percentages = sorted(match_percentages)
worst_match_percentage = sorted_percentages[0]  # Worst performance
percentile_1st = np.percentile(sorted_percentages, 1)  # 1st percentile performance

print(f"BENCHMARK COMPLETE\n")
print(f"Perfect sequence predictions: {accuracy_percentage:.2f}%")
print(f"Prediced character correctly: {average_match_percentage:.2f}%")
print(f"Worst sequence: {worst_match_percentage:.2f}%")
print(f"1st percentile sequence: {percentile_1st:.2f}%")

output_file_path = 'benchmark/results_summary.txt'
with open(output_file_path, 'w') as file:
    file.write(f"Perfect sequence predictions: {accuracy_percentage:.2f}%\n")
    file.write(f"Prediced character correctly: {average_match_percentage:.2f}%\n")
    file.write(f"Worst sequence: {worst_match_percentage:.2f}%\n")
    file.write(f"1st percentile sequence: {percentile_1st:.2f}%\n")