import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
import re
from sklearn.cluster import DBSCAN
from collections import Counter
import torch



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

def clean_image(image, eps=10, min_points_per_cluster=500):
    """
    Remove small clusters from an image using DBSCAN.

    Parameters:
    - image (PIL image):  Image to be cleaned.
    - eps (float): The maximum distance between two samples for one to be
      considered as in the neighborhood of the other in DBSCAN.
    - min_points_per_cluster (int): Minimum number of points required for a
      cluster to be considered valid. Clusters with fewer points will be removed.

    Returns:
    - numpy.ndarray: Cleaned binary image with small clusters removed.
    """

    # Convert the image to a NumPy array
    data = np.array(image)

    # Find the coordinates of non-zero pixels
    x_cor, y_cor = np.where(data > 0)

    # Create a feature matrix from the coordinates
    X = np.array([[x, y] for x, y in zip(x_cor, y_cor)])

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps)
    labels = dbscan.fit_predict(X)

    # Count the number of points in each cluster
    points_per_cluster = Counter(labels)

    # Identify clusters with fewer points than the threshold
    clusters_to_drop = [cluster for cluster in points_per_cluster if points_per_cluster[cluster] < min_points_per_cluster]

    # Create a mask to exclude points in small clusters
    mask = np.isin(labels, clusters_to_drop, invert=True)

    # Create a cleaned image by applying the mask
    clean_data = np.zeros_like(data)
    clean_data[x_cor[mask], y_cor[mask]] = 255
    return clean_data

def create_filename(length_loader, index):
    """
    Generate a filename based on the given length_loader and index.

    The generated filename follows a specific pattern: 'prediction_{index}.png'.
    If the length_loader is greater than 1000, the index will be formatted with
    leading zeros to ensure a consistent filename length.

    Parameters:
    - length_loader (int): The length of the loader or dataset.
    - index (int): The index used to generate the filename.

    Returns:
    - str: The generated filename.
    """
    if length_loader > 1000:
        return f'prediction_{index + 1:04d}.png'
    return f'prediction_{index + 1:03d}.png'

def get_label(output, proba_threshold):
    """
    Convert a probability output to binary labels based on a specified threshold.

    The function takes a probability output tensor and converts it to binary labels
    using the provided probability threshold. If the probability at a particular position
    in the output tensor is greater than the threshold, the corresponding label is set
    to 1; otherwise, it is set to 0.

    Parameters:
    - output (torch.Tensor): The probability output tensor.
    - proba_threshold (float): The probability threshold for binary labeling.

    Returns:
    - torch.Tensor: Binary labels tensor based on the probability threshold.
    """
    return (output > proba_threshold).type(torch.uint8)

def save_prediction(output, filename):
    """
    Save a prediction output as an image file.

    The function takes a prediction output tensor, scales it to the range [0, 255],
    converts it to a NumPy array, and saves it as an image file with the specified filename.

    Parameters:
    - output (torch.Tensor): The prediction output tensor.
    - filename (str): The filename (including path) to save the image.

    Returns:
    - None: The function does not return any value but saves the image to the specified filename.
    """
    # Squeeze the output tensor and scale it to the range [0, 255]
    pred = torch.squeeze(output * 255).cpu().numpy()

    # Create a PIL Image from the NumPy array
    img = Image.fromarray(pred)

    # Save the image to the specified filename
    img.save(filename)