import matplotlib.image as mpimg
import numpy as np
import os
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import re

def load_images_from_folder(folder):
    """
    Load all images from a specified folder.

    Parameters:
    - folder (str): The path to the folder containing images.

    Returns:
    - list: A list of NumPy arrays, each representing an image.

    Example:
    >>> training_data_folder = "path/to/your/data/training/training_data"
    >>> groundtruth_folder = "path/to/your/data/training/groundtruth"
    >>> training_data_images = load_images_from_folder(training_data_folder)
    >>> groundtruth_images = load_images_from_folder(groundtruth_folder)
    """
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(folder, filename)
            image = mpimg.imread(filepath)
            images.append(image)
    return images


def img_crop(im, w, h):
    """
    Crop an input image into smaller patches.

    Parameters:
    - im (numpy.ndarray): Input image represented as a NumPy array.
    - w (int): Width of the patches.
    - h (int): Height of the patches.

    Returns:
    - list: A list of cropped patches, where each element is a NumPy array representing a patch.

    Example:
    >>> input_image = np.random.rand(256, 256, 3)
    >>> width, height = 64, 64
    >>> cropped_patches = img_crop(input_image, width, height)
    """
 
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches

def img_float_to_uint8(img):
    """
    Convert a floating-point image to 8-bit unsigned integer format.

    Parameters:
    - img (numpy.ndarray): Input image represented as a NumPy array.

    Returns:
    - numpy.ndarray: Output image with pixel values scaled to the range [0, 255] and converted to uint8.

    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def value_to_class(v, foreground_threshold):
    """
    Convert a value to a binary class based on a foreground threshold.

    Parameters:
    - v (numpy.ndarray): Input value or array.
    - foreground_threshold (float): Threshold for determining the binary class.

    Returns:
    - int: Binary class (0 or 1).
    """
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
def load_img_for_vit(directory):
    """
    Load images from a specified directory.

    Args:
        directory (str): The path to the directory containing images.

    Returns:
        list: A list of loaded images.
    """

    # Get a list of image file paths
    image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(('.png', '.jpg', '.jpeg'))]

    # Load images
    images = [Image.open(image_path) for image_path in image_paths]
    
    return images
def patch_to_label(patch):
    df = np.mean(patch)
    if df > 0.25:
        return 1
    else:
        return 0
def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))