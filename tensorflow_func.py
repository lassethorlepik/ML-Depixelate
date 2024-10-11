import os
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import tensorflow as tf
import keras
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import ops, layers, regularizers
import Levenshtein
# Internal tools
from util import generate_random_string


def split_data(images, labels, train_size=0.9, shuffle=False):
    """Split the images and labels into training and validation datasets.
    
    Example: split_data(images, labels, train_size=0.9, shuffle=False)
    
    If total dataset Size: 1000 then
    
    Training Indices: [0, 1, 2, ..., 899]
    Validation Indices: [900, 901, ..., 999]
    """
    
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = tf.keras.ops.arange(size)
    if shuffle:
        keras.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


def add_noise(img, noise_factor=1):
    # Adding random noise to the image
    if tf.random.uniform(()) < 0.5:
        return img
    noise = noise_factor * tf.random.uniform((), 0.0, 0.002)
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=noise, dtype=tf.float32)
    img = img + noise
    img = tf.clip_by_value(img, 0.0, 1.0)  # Make sure the image values are still in [0, 1] range
    return img


def apply_compression(img, quality=70):
    # Simulate JPEG compression by encoding and then decoding the image
    if tf.random.uniform(()) < 0.5:
        return img
    img = tf.image.encode_jpeg(tf.image.convert_image_dtype(img, tf.uint8, saturate=True), quality=quality)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)  # Convert back to float32
    return img


def random_shift(img, max_shift=3):
    # Randomly generate the number of pixels to pad on the top and left
    pad_top = tf.random.uniform(shape=(), minval=0, maxval=max_shift+1, dtype=tf.int32)
    pad_left = tf.random.uniform(shape=(), minval=0, maxval=max_shift+1, dtype=tf.int32)
    
    # Pad the image with white (value=1) on the top and left
    img_padded = tf.pad(img, paddings=[[pad_top, 0], [pad_left, 0], [0, 0]], mode="CONSTANT", constant_values=1)
    
    # Determine the new height and width after padding
    new_height = tf.shape(img)[0] + pad_top
    new_width = tf.shape(img)[1] + pad_left
    
    # Crop the padded image to the original size from the bottom right corner to shift it down and to the right
    img_shifted = tf.image.crop_to_bounding_box(img_padded, 0, 0, new_height - pad_top, new_width - pad_left)
    
    return img_shifted


def detect_block_size_tf(image):
    """
    Detect block size in a TensorFlow compatible way.
    """
    # Assuming image is a 2D tensor (grayscale)
    
    # Calculate the difference between adjacent pixels along the width
    pixel_diffs = tf.abs(image[:, :-1] - image[:, 1:])
    
    # Detect changes
    change_indices = tf.where(pixel_diffs > 0)
    
    if tf.shape(change_indices)[0] == 0:
        # No change found, return default block size or image width
        # Explicitly cast the return value to tf.int32
        return tf.cast(tf.shape(image)[1], tf.int32)
    else:
        # Use the first change as an approximation of block size
        # Also explicitly cast this return value to tf.int32
        first_change = change_indices[0][1]
        return tf.cast(first_change + 1, tf.int32)


def pad_custom_color(image, target_width, target_height, pad_color=(1, 1, 1)):
    """
    Pad the given image on the right and bottom to the target dimensions with the specified color.

    Parameters:
    - image: A 3D tensor of shape [height, width, channels].
    - target_width: Desired width after padding.
    - target_height: Desired height after padding.
    - pad_color: Padding color, default is white (1, 1, 1) for images in [0, 1] range.

    Returns:
    - Padded image tensor.
    """
    # Calculate the padding sizes
    pad_height = target_height - tf.shape(image)[0]
    pad_width = target_width - tf.shape(image)[1]
    
    # Ensure padding is non-negative
    pad_height = tf.maximum(0, pad_height)
    pad_width = tf.maximum(0, pad_width)
    
    # Pad the image
    padded_image = tf.pad(
        image, 
        paddings=[[0, pad_height], [0, pad_width], [0, 0]], 
        mode='CONSTANT', 
        constant_values=pad_color[0]  # Assuming uniform padding color for all channels
    )
    
    return padded_image


def downscale_image(img, block_size, img_width, img_height):
    # Get original dimensions
    original_height, original_width = tf.cast(tf.shape(img)[0], tf.float32), tf.cast(tf.shape(img)[1], tf.float32)
    # Calculate new dimensions based on the block size
    downscaled_height = original_height / tf.cast(block_size, tf.float32)
    downscaled_width = original_width / tf.cast(block_size, tf.float32)
    
    # Downscale using nearest neighbor interpolation
    downscaled_img = tf.image.resize(img, [tf.cast(downscaled_height, tf.int32), tf.cast(downscaled_width, tf.int32)], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    # If image is still larger than model supports, then scale it down by force, keep aspect ratio
    scale_height = tf.cast(img_height, tf.float32) / downscaled_height
    scale_width = tf.cast(img_width, tf.float32) / downscaled_width
    scale = tf.minimum(scale_height, scale_width)
    
    # Apply additional scaling if necessary
    if scale < 1.0:
        new_height = tf.cast(downscaled_height * scale, tf.int32)
        new_width = tf.cast(downscaled_width * scale, tf.int32)
        resized_img = tf.image.resize(downscaled_img, [new_height, new_width], method=tf.image.ResizeMethod.BICUBIC)
    else:
        resized_img = downscaled_img
    
    return resized_img


def print_dimensions(image, label):
    # Print the dimensions of the image
    new_height, new_width = tf.shape(image)[0], tf.shape(image)[1]
    tf.print(f"Padded image dimensions: {new_width}x{new_height}")
    return image, label


def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    label_length = ops.cast(ops.squeeze(label_length, axis=-1), dtype="int32")
    input_length = ops.cast(ops.squeeze(input_length, axis=-1), dtype="int32")
    sparse_labels = ops.cast(
        ctc_label_dense_to_sparse(y_true, label_length), dtype="int32"
    )

    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())

    return ops.expand_dims(
        tf.compat.v1.nn.ctc_loss(
            inputs=y_pred, labels=sparse_labels, sequence_length=input_length
        ),
        1,
    )


def ctc_label_dense_to_sparse(labels, label_lengths):
    label_shape = ops.shape(labels)
    num_batches_tns = ops.stack([label_shape[0]])
    max_num_labels_tns = ops.stack([label_shape[1]])

    def range_less_than(old_input, current_input):
        return ops.expand_dims(ops.arange(ops.shape(old_input)[1]), 0) < tf.fill(
            max_num_labels_tns, current_input
        )

    init = ops.cast(tf.fill([1, label_shape[1]], 0), dtype="bool")
    dense_mask = tf.compat.v1.scan(
        range_less_than, label_lengths, initializer=init, parallel_iterations=1
    )
    dense_mask = dense_mask[:, 0, :]

    label_array = ops.reshape(
        ops.tile(ops.arange(0, label_shape[1]), num_batches_tns), label_shape
    )
    label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)

    batch_array = ops.transpose(
        ops.reshape(
            ops.tile(ops.arange(0, label_shape[0]), max_num_labels_tns),
            tf.reverse(label_shape, [0]),
        )
    )
    batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
    indices = ops.transpose(
        ops.reshape(ops.concatenate([batch_ind, label_ind], axis=0), [2, -1])
    )

    vals_sparse = tf.compat.v1.gather_nd(labels, indices)

    return tf.SparseTensor(
        ops.cast(indices, dtype="int64"), 
        vals_sparse, 
        ops.cast(label_shape, dtype="int64")
    )


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    input_shape = ops.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())
    input_length = ops.cast(input_length, dtype="int32")

    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length
        )
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    return (decoded_dense, log_prob)


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(ops.shape(y_true)[1], dtype="int64")

        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_model(img_width, img_height, char_to_num):
    # Inputs to the model
    input_img = layers.Input(
        shape=(img_height, img_width, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # Define the L2 regularization factor
    l2_regularizer = regularizers.l2(0.001)
    
    x = layers.Conv2D(
        32,
        (3, 3),
        kernel_regularizer=l2_regularizer,
        kernel_initializer="he_normal",
        activation="relu",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.Conv2D(
        64,
        (3, 3),
        kernel_regularizer=l2_regularizer,
        kernel_initializer="he_normal",
        activation="relu",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.Conv2D(
        128,
        (3, 3),
        kernel_regularizer=l2_regularizer,
        kernel_initializer="he_normal",
        activation="relu",
        padding="same",
        name="Conv3",
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Permute((2, 1, 3), name="permute")(x)
    new_shape = (x.shape[1], x.shape[2] * 128)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(x)

    # Output layer, + 1 required for blank token
    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense_output"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs={"image": input_img, "label": labels}, outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=0.001)
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


# A utility function to decode the output of the network
def decode_batch_predictions(pred, max_length, num_to_char):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search
    results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


class PredictionVisualizer(keras.callbacks.Callback):
    def __init__(self, model, validation_data, max_length, num_to_char):
        self.prediction_model = model
        self.validation_data = validation_data
        self.max_length = max_length
        self.num_to_char = num_to_char
        self.losses = []  # Initialize an empty list to store loss values

    def on_epoch_end(self, epoch, logs=None):
        # Append the loss value for the current epoch to the list
        if logs is not None:
            self.losses.append(logs.get('loss'))
        # Select a batch from the validation dataset
        for batch in self.validation_data.take(1):
            batch_images = batch["image"]
            batch_labels = batch["label"]
            
            preds = self.prediction_model.predict(batch_images)
            pred_texts = decode_batch_predictions(preds, self.max_length, self.num_to_char)
            
            orig_texts = []
            for label in batch_labels:
                label = tf.strings.reduce_join(self.num_to_char(label)).numpy().decode("utf-8")
                orig_texts.append(label)
            
            n_rows = (len(orig_texts) + 3) // 4
            _, ax = plt.subplots(n_rows, 4, figsize=(10, n_rows * 1.25))
            for i in range(len(pred_texts)):
                # Upscale the image by a factor of 4 using nearest neighbor interpolation
                img = batch_images[i]
                upscaled_img = tf.image.resize(img, [img.shape[0] * 4, img.shape[1] * 4], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                upscaled_img = (upscaled_img * 255).numpy().astype(np.uint8)
                pred_t = pred_texts[i].replace("[UNK]", "")
                orig_t = orig_texts[i].replace("[UNK]", "")
                match_percentage = similarity(pred_t, orig_t)
                is_correct = pred_t == orig_t
                color = 'green' if is_correct else 'red'
                title = f"Pred: {pred_t}\nTrue: {orig_t}\nMatch: {match_percentage:.2f}%"
                
                # Display the image
                ax[i // 4, i % 4].imshow(upscaled_img[:, :, 0], cmap="gray")
                ax[i // 4, i % 4].set_title(title, fontsize=8, color=color)
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
            plt.savefig(f'results/epoch_{epoch+1}_{generate_random_string(4)}.png')
            plt.close()
            
            # Plotting loss history after the last epoch
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(self.losses) + 1), self.losses, label='Training Loss')
            plt.title('Epochs vs. Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.xticks(np.arange(1, len(self.losses) + 1, step=max(1, len(self.losses) // 10)))
            # Ensure x-axis labels match epoch numbers
            plt.legend()
            plt.grid(True)
            plt.savefig(f'results/loss_history_{epoch+1}_{generate_random_string(4)}.png')
            plt.close()


def similarity(str1, str2):
    """Evaluate string similarity."""
    # TODO: Use a better, more balanced metric
    
    # Calculate Levenshtein distance
    distance = Levenshtein.distance(str1, str2)
    # Normalize the distance to get a similarity measure
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 100  # Both strings are empty
    similarity_score = (1 - distance / max_len) * 100
    return similarity_score