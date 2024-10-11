import datetime
import builtins
import os
import platform
import random
import string
from colorama import init, Fore
import tensorflow as tf

init() # Initializes colorama to make ANSI escape character sequences work under MS Windows as well


def print(text):
    "Print string as a fancy timestamped log."
    # Get current time with microseconds
    now = datetime.datetime.now()
    # Format time as hh:mm:ss:ms
    formatted_time = now.strftime("%H:%M:%S") + f':{now.microsecond // 1000:03d}'
    # Print the text with yellow color for the timestamp
    builtins.print(f"{Fore.YELLOW}[{formatted_time}]{Fore.RESET} {text}")


def remove_prefix(text):
    """Remove the prefix."""
    split = text.split("_")
    if (len(split) > 1):
        return split[1]
    else:
        return text


def generate_random_string(length):
    """Generate a random string of letters and numbers with fixed length."""
    # Combine letters and digits
    characters = string.ascii_letters + string.digits
    # Randomly choose characters and join them to form a string
    random_string = ''.join(random.choice(characters) for i in range(length))
    return random_string


def check_gpu_compute():
    # WINDOWS HAS NO GPU SUPPORT AFTER TF 2.10, WLS REQUIRED
    if not tf.test.is_built_with_cuda():
        builtins.print("Tensorflow build does not support CUDA")
    else:
        build_info = tf.sysconfig.get_build_info()
        cuda_version = build_info["cuda_version"]
        cudnn_version = build_info["cudnn_version"]
        builtins.print(f"Tensorflow version: {tf.__version__} built for CUDA {cuda_version} and cuDNN: {cudnn_version}")
    builtins.print(f"Python version: {platform.python_version()}")
    builtins.print(f"Available devices: {tf.config.list_physical_devices()}")
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth on GPUs to avoid allocation issues
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            builtins.print("Using GPU:", gpus)
        except RuntimeError as e:
            builtins.print(e)
    else:
        builtins.print("No GPU found, using CPU instead.")


def setup_folders():
    os.makedirs("results", exist_ok=True)
    os.makedirs("benchmark", exist_ok=True)