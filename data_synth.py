import glob
import os
import random
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool, cpu_count
#import cProfile
#import pstats


# SETTINGS =======================
# Size of the generated dataset
N_IMAGES = 50000
# All characters used in random strings
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# Shortest string
MIN_LEN = 1
# Longest string
MAX_LEN = 20
# Size of the original image
X = 512
Y = 64
# Smallest pixelation block size range
MIN_BLOCK = 8
MAX_BLOCK = 8
# Font size range
MIN_FONT = 24
MAX_FONT = 28
# Top and left padding for text in the original image
MIN_PAD = 0
MAX_PAD = 20
# Fonts used on image generation (random choice from set every image)
FONTS = [
        "arial.ttf",
    ]

"""
    "arialbd.ttf",
    "ariali.ttf",
    "verdana.ttf",
    "times.ttf",
    "cour.ttf",
    "georgia.ttf",
    "trebuc.ttf",
    "comic.ttf",
    "impact.ttf",
    "calibri.ttf",
    "cambria.ttc"
"""

# ================================


# Dimensions of saved images
FINAL_X = X // MAX_BLOCK
FINAL_Y = Y // MAX_BLOCK

parameters = [
    str(X), str(Y),
    str(MIN_LEN), str(MAX_LEN), str(MIN_BLOCK), str(MAX_BLOCK),
    str(MIN_FONT), str(MAX_FONT), str(MIN_PAD), str(MAX_PAD)
]

name = f"N{N_IMAGES}-" + "-".join(parameters) + f"-F{len(FONTS)}-{FONTS[0]}"


def main():
    debug_folder = f"datasets/debug-{name}"
    os.makedirs(debug_folder, exist_ok=True)
    create_debug_images(debug_folder)

    output_folder = f"datasets/dataset-{name}"
    os.makedirs(output_folder, exist_ok=True)
    delete_png_files(output_folder)

    num_processes = cpu_count()
    images_per_process = N_IMAGES // num_processes
    ranges = [(i * images_per_process, (i + 1) * images_per_process) for i in range(num_processes)]
    ranges[-1] = (ranges[-1][0], N_IMAGES)  # Adjust the last range
    # Multiprocess
    with Pool(processes=num_processes) as pool:
        tasks = [(start, end, output_folder, i == 0) for i, (start, end) in enumerate(ranges)]
        pool.starmap(create_images_for_range, tasks)


def create_images_for_range(start_index, end_index, output_folder, report_progress=False):
    """Create images for a specified range and optionally report progress."""
    total_images = end_index - start_index
    progress_report_interval = max(1, total_images // 100)
    for i in range(start_index, end_index):
        create_image(i, output_folder)
        if report_progress and (i - start_index) % progress_report_interval == 0:
            progress_percent = ((i - start_index) / total_images) * 100
            print(f"Progress: {progress_percent:.1f}%")


def create_image(index, output_folder):
    """Generate and save an image for the dataset."""
    length = random.randint(MIN_LEN, MAX_LEN)
    text = generate_random_string(length)
    block_size = random.randint(MIN_BLOCK, MAX_BLOCK)
    font_size = random.randint(MIN_FONT, MAX_FONT)
    font = ImageFont.truetype(random.choice(FONTS), font_size)

    # Draw original image
    background_color = 255
    text_color = 0
    bitmap = Image.new('L', (X, Y), background_color)
    draw = ImageDraw.Draw(bitmap)
    draw.fontmode = "L" if random.choice([True, False]) else "1" # L = anti-aliasing, 1 = no anti-aliasing
    draw.text((random.randint(MIN_PAD, MAX_PAD), random.randint(MIN_PAD, MAX_PAD)), text, font=font, fill=text_color)
    
    # Pixelate, shift pattern up to MAX_BLOCK
    pixelate_image_mono(bitmap, block_size, random.randint(-MAX_BLOCK, MAX_BLOCK), random.randint(-MAX_BLOCK, MAX_BLOCK))
    bitmap = crop_and_pad_image(bitmap, block_size, background_color)
    
    # Save file
    file_name = f"{output_folder}/{index}_{text}.png"
    bitmap.save(file_name)


def pixelate_image_mono(image, block_size, shift_x=0, shift_y=0):
    """Pixelate given image using block averaging, only supports monochrome."""
    width, height = image.size
    pixels = image.load()

    # Adjust shifts to keep them within the image boundaries
    shift_x = shift_x % block_size
    shift_y = shift_y % block_size

    # Loop over each block
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            intensity_sum, count = 0, 0

            # Calculate the start and end points considering the shift
            y_start = y + shift_y
            x_start = x + shift_x

            # Accumulate sum of pixels within the block, using 255 for out-of-bound values
            for iy in range(block_size):
                for ix in range(block_size):
                    pos_x = x_start + ix
                    pos_y = y_start + iy

                    # Check if the position is within bounds, use 255 if out-of-bounds
                    if 0 <= pos_x < width and 0 <= pos_y < height:
                        intensity = pixels[pos_x, pos_y]
                    else:
                        intensity = 255  # Use white for out-of-bounds pixels

                    intensity_sum += intensity
                    count += 1

            # Calculate average intensity
            if count > 0:
                avg_intensity = intensity_sum // count

            # Assign average intensity to pixels within the block
            for iy in range(block_size):
                for ix in range(block_size):
                    pos_x = x + ix
                    pos_y = y + iy
                    if 0 <= pos_x < width and 0 <= pos_y < height:  # Only assign within image bounds
                        pixels[pos_x, pos_y] = avg_intensity


def find_bounding_box(image):
    """Find the smallest rectangle to encompass all pixels different than background."""
    background_color = 255
    width, height = image.size
    left, top, right, bottom = width, height, -1, -1
    pixels = image.load()

    # Find actual bounds
    for y in range(height):
        for x in range(width):
            if pixels[x, y] != background_color:
                if x < left: left = x
                if x > right: right = x
                if y < top: top = y
                if y > bottom: bottom = y

    # Return bounds, handle cases where no pixels are different from the background
    if right == -1 or bottom == -1:  # Indicates no non-background pixels were found
        return (0, 0, width, height)
    return (left, top, right, bottom)


def crop_and_pad_image(image, block_size, background_color):
    """Resize image according to the block size, so that each block is one pixel to save space, pad extra space with background_color."""
    # Crop out empty space
    bbox = find_bounding_box(image)
    cropped_image = image.crop(bbox)
    
    # Ensures that if there's any remainder when dividing cropped_image.width by block_size,
    # the result rounds up to include the extra block needed to cover the entire width.
    new_width = (cropped_image.width + block_size - 1) // block_size
    new_height = (cropped_image.height + block_size - 1) // block_size
    scaled_image = cropped_image.resize((new_width, new_height), Image.NEAREST)
        
    # Create white image and paste the pixelated image top left
    padded_image = Image.new('L', (FINAL_X, FINAL_Y), background_color)
    padded_image.paste(scaled_image, (0, 0))

    return padded_image


def generate_random_string(length):
    """Generate a random string with specified length."""
    return ''.join(random.choice(CHARSET) for _ in range(length))


def delete_png_files(folder_path):
    """Delete all images in a folder."""
    # Construct the path pattern to match all PNG files in the folder
    pattern = os.path.join(folder_path, '*.png')
    # Use glob to find all files matching the pattern
    png_files = glob.glob(pattern)
    # Loop over the list of file paths & remove each file
    for file_path in png_files:
        os.remove(file_path)


def create_debug_images(output_folder):
    """Create special images for debugging."""
    delete_png_files(output_folder)
    text = generate_random_string(MAX_LEN)
    background_color = 255
    text_color = 0
    
    block_size = MAX_BLOCK
    font_size = MAX_FONT
    font = ImageFont.truetype(random.choice(FONTS), font_size)
    bitmap = Image.new('L', (X, Y), background_color)
    draw = ImageDraw.Draw(bitmap)
    draw.text((random.randint(MIN_PAD, MAX_PAD), random.randint(MIN_PAD, MAX_PAD)), text, font=font, fill=text_color)
    file_name = f"{output_folder}/txt_max_{text}.png"
    bitmap.save(file_name)
    
    pixelate_image_mono(bitmap, block_size, shift_x=0, shift_y=0)
    #bbox = find_bounding_box(bitmap)
    #draw.rectangle(bbox, outline='red', width=3)  # Draw a red bounding box
    file_name = f"{output_folder}/pix_max_{text}.png"
    bitmap.save(file_name)
    
    bitmap = crop_and_pad_image(bitmap, block_size, background_color)
    file_name = f"{output_folder}/crp_max_{text}.png"
    bitmap.save(file_name)
    
    block_size = MIN_BLOCK
    font_size = MIN_FONT
    font = ImageFont.truetype(random.choice(FONTS), font_size)
    bitmap = Image.new('L', (X, Y), background_color)
    draw = ImageDraw.Draw(bitmap)
    draw.text((random.randint(MIN_PAD, MAX_PAD), random.randint(MIN_PAD, MAX_PAD)), text, font=font, fill=text_color)
    file_name = f"{output_folder}/txt_min_{text}.png"
    bitmap.save(file_name)
    
    pixelate_image_mono(bitmap, block_size, shift_x=-10, shift_y=10)
    #bbox = find_bounding_box(bitmap)
    #draw.rectangle(bbox, outline='red', width=1)  # Draw a red bounding box
    file_name = f"{output_folder}/pix_min_{text}.png"
    bitmap.save(file_name)
    
    bitmap = crop_and_pad_image(bitmap, block_size, background_color)
    file_name = f"{output_folder}/crp_min_{text}.png"
    bitmap.save(file_name)


if __name__ == "__main__":
    #profiler = cProfile.Profile()
    #profiler.enable()
    main()
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('cumtime')
    #stats.print_stats()



# ===========================
# UNUSED
# ===========================



def color_distance(c1, c2):
    """Calculate the Euclidean distance between two colors in RGB space."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def generate_random_color():
    # Generate a random number between 0 and 1
    random_chance = random.random()
    # 30% chance of white
    if random_chance < 0.30:
        return (255, 255, 255)  # White
    # 30% chance of black
    elif random_chance < 0.60:
        return (0, 0, 0)  # Black
    # Remaining 40% chance, generate a totally random color
    else:
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def get_colors(min_distance=60):
    """Generate two colors with a minimum distance to avoid similar colors."""
    background_color = generate_random_color()
    text_color = generate_random_color()
    while color_distance(background_color, text_color) < min_distance:
        text_color = generate_random_color()
    return background_color, text_color


def find_bounding_box_by_color(image, background_color):
    width, height = image.size
    left, top, right, bottom = width, height, -1, -1
    pixels = image.load()

    # Find actual bounds
    for y in range(height):
        for x in range(width):
            if pixels[x, y] != background_color:
                if x < left: left = x
                if x > right: right = x
                if y < top: top = y
                if y > bottom: bottom = y

    # Return bounds, handle cases where no pixels are different from the background
    if right == -1 or bottom == -1:  # No change in initial right and bottom values means no pixels found
        return (0, 0, width, height)
    return (left, top, right, bottom)
