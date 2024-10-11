
import os
from keras import layers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# Internal tools
from tensorflow_func import add_noise, apply_compression, detect_block_size_tf, downscale_image, pad_custom_color, random_shift, split_data
from util import remove_prefix, print


class DatasetInstance():
    def __init__(self, data_dir, batch_size, img_width, img_height, train_data_amount):
            self.data_dir = data_dir
            self.batch_size = batch_size
            self.img_width = img_width
            self.img_height = img_height
            self.train_data_amount = train_data_amount
            self.load_dataset()
            self.preprocess()
            self.dataset_finalize()
    
    def load_dataset(self):
        # Get list of all the images
        self.images = sorted(list(map(str, list(self.data_dir.glob("*.png")))))
        self.labels = [remove_prefix(img.split(os.path.sep)[-1].split(".png")[0]) for img in self.images]
        characters = set(char for label in self.labels for char in label)
        self.characters = sorted(list(characters))

        print(f"Number of images found: {len(self.images)}")
        print(f"Number of labels found: {len(self.labels)}")
        print(f"Number of unique characters: {len(characters)}")
        print(f"Characters present: {characters}")
        
        # Maximum length of any string in the dataset
        self.max_length = max([len(label) for label in self.labels])
        
    def preprocess(self):
        # Mapping characters to integers
        self.char_to_num = layers.StringLookup(vocabulary=list(self.characters), mask_token=None)
        # Mapping integers back to original characters
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
        print("Splitting data into training and validation sets...")
        # Do not shuffle to get deterministic groups of training and validation sets (otherwise we compromise validity of a validation set)
        self.x_train, self.x_valid, self.y_train, self.y_valid = split_data(np.array(self.images), np.array(self.labels), self.train_data_amount)
    
    def encode_single_sample(self, img_path, label):
        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Downscale so that each block is a pixel
        #img = downscale_image(img, detect_block_size_tf(img), self.img_width, self.img_height)
        # 5. Pad to uniform dimensions
        #img = pad_custom_color(img, self.img_width, self.img_height)
        # tf.py_function(func=print_dimensions, inp=[img, label], Tout=[tf.float32, tf.string])  # Debugging
        # 5. Apply noise
        #img = add_noise(img)
        # 6. Apply compression
        #img = apply_compression(img)
        # 7. Random shifts
        #img = random_shift(img)
        # 9. Map the characters in label to numbers
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        # 10. Return a dict as our model is expecting two inputs
        return {"image": img, "label": label}
    
    def dataset_finalize(self):
        # Create Dataset objects
        pad_token = 0
        padded_shapes = {
            'image': (self.img_height, self.img_width, 1),  # Assuming all images already have the same size
            'label': [None]  # Padding the sequence dimension of the labels
        }
        padding_values = {
            'image': 0.0,  # Assuming images are normalized to [0, 1], use 0 for padding
            'label': tf.constant(pad_token, dtype=tf.int64)  # Use a pad token for labels
        }

        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.train_dataset = (
            train_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .padded_batch(self.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        validation_dataset = tf.data.Dataset.from_tensor_slices((self.x_valid, self.y_valid))
        self.validation_dataset = (
            validation_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .padded_batch(self.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        
    def visualize_training_dataset(self):
        """Show a popup with images from the training dataset."""
        print("Displaying examples from dataset for user validation...")
        _, ax = plt.subplots(4, 4, figsize=(10, 5))
        for batch in self.train_dataset.take(1):
            images = batch["image"]
            labels = batch["label"]
            for i in range(min(self.batch_size, 16)):
                # Upscale the image by a factor of 4 using nearest neighbor interpolation
                img = images[i]
                upscaled_img = tf.image.resize(img, [img.shape[0] * 4, img.shape[1] * 4], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                upscaled_img = (upscaled_img * 255).numpy().astype(np.uint8)
                
                label = tf.strings.reduce_join(self.num_to_char(labels[i])).numpy().decode("utf-8").replace("[UNK]", "")

                ax[i // 4, i % 4].imshow(upscaled_img[:, :, 0], cmap="gray")
                ax[i // 4, i % 4].set_title(label, fontsize=8)
                ax[i // 4, i % 4].axis("off")
                
                # Get the axis limits
                xlim = ax[i // 4, i % 4].get_xlim()
                ylim = ax[i // 4, i % 4].get_ylim()
                
                # Calculate rectangle dimensions
                rect_width = xlim[1] - xlim[0]
                rect_height = ylim[0] - ylim[1]  # y-axis is inverted in matplotlib
                
                # Create a Rectangle patch with no fill to outline the image
                rect = Rectangle((xlim[0], ylim[1]), rect_width, rect_height, linewidth=1, edgecolor='black', facecolor='none')
                
                # Add the rectangle to the Axes
                ax[i // 4, i % 4].add_patch(rect)
                
        plt.tight_layout()
        plt.show()