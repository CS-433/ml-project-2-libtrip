import matplotlib.image as mpimg
import numpy as np
import os
import matplotlib.image as mpimg
import numpy as np

IMG_PATCH_SIZE = 16


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
        if filename.endswith((".png", ".jpg", ".jpeg")):
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
    foreground_threshold = 0.25
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")


def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(gt_imgs)
    gt_patches = [
        img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = np.asarray(
        [
            gt_patches[i][j]
            for i in range(len(gt_patches))
            for j in range(len(gt_patches[i]))
        ]
    )
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)