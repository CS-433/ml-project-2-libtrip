import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
import re
from sklearn.cluster import DBSCAN
from collections import Counter
import torch
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset, random_split
from sklearn.metrics import accuracy_score, f1_score

def split_data(dataset,test_ratio,seed):
    
    """Splits a dataset into random train and test subsets.

    Args:
        dataset (Dataset): dataset.
        test_ratio (float): test proportion (between 0 and 1).
        seed (int, optional): seed. Defaults to None.

    Returns:
        Tuple[Dataset, Dataset]: train and test datasets.
    """
    # Define generator
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    # Define lengths of subsets
    train_ratio = 1 - test_ratio
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    lengths = [train_size, test_size]

    # Split
    train_dataset, test_dataset = random_split(dataset, lengths, generator)

    return train_dataset, test_dataset

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

def clean_(data,blackOrWhite,eps=10,min_samples=500):
    dbscan = DBSCAN(eps=eps)
    if blackOrWhite:
        x_cor, y_cor = np.where(data == 0)
    else:
        x_cor, y_cor = np.where(data >0)
    X = np.array([[x, y] for x, y in zip(x_cor, y_cor)])
    labels= dbscan.fit_predict(X)
    points_per_cluster = Counter(labels)
    clusters_to_drop = [cluster for cluster in points_per_cluster if points_per_cluster[cluster] < min_samples]
    mask = np.isin(labels, clusters_to_drop, invert=True)
    
    if blackOrWhite:
        clean_data = np.full_like(data, 255)
        clean_data[x_cor[mask], y_cor[mask]] = 0
    else:
        clean_data = np.full_like(data, 0)
        clean_data[x_cor[mask], y_cor[mask]] = 255
    
    return clean_data
    

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
    data= np.array(image)
    first_process= clean_(data,True)
    return first_process#clean_(first_process,False)

def process_images(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png'))]

    # Process each image
    for image_file in image_files:
        # Load the image
        input_path = os.path.join(input_folder, image_file)
        image = Image.open(input_path)

        # Clean the image
        cleaned_image = clean_image(image)

        # Save the cleaned image to the output folder
        output_path = os.path.join(output_folder, image_file)
        Image.fromarray(cleaned_image).save(output_path)

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
    
    
def accuracy_(target, output):
    """Accuracy classification score from tensors.

    Args:
        target (torch.Tensor): Ground truth (correct) labels.
        output (torch.Tensor): Predicted labels, as returned by a classifier.

    Returns:
        float: accuracy score between 0 and 1.
    """
    target_flatten = torch.flatten(target).cpu()
    output_flatten = torch.flatten(output).cpu()
    return accuracy_score(target_flatten, output_flatten, normalize=True)


def f1_(target, output):
    """F1 score from tensors.

    Args:
        target (torch.Tensor): Ground truth (correct) labels.
        output (torch.Tensor): Predicted labels, as returned by a classifier.

    Returns:
        float: f1 score between 0 and 1.
    """
    target_flatten = torch.flatten(target).cpu()
    output_flatten = torch.flatten(output).cpu()
    return f1_score(target_flatten, output_flatten, zero_division=1)

def train(model, train_loader, criterion, optimizer, device,epochs, lr_scheduler=None):
    model.train()
    losses= list()
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        # Iterate over the training data
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch'):
            # Send the input to the device
            data, target = data.to(device), target.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update the total loss
            total_loss += loss.item()

        # Average loss for the epoch
        average_loss = total_loss / len(train_loader)
        print(f"average loss: ", average_loss)
        losses.append((epoch,average_loss))

        # Adjust learning rate if a scheduler is provided
        if lr_scheduler is not None:
            lr_scheduler.step(average_loss)
    return losses

def predict(model, test_loader, device,pred_path,threshhold=0.25):
    model.eval()
    f1_scores = list()
    accuracy_scores = list()
    prediction_filnames = list()
    with torch.no_grad():
    # Loop over the dataset
        for i, (data, target) in enumerate(test_loader):
            filename = create_filename(len(test_loader),i)
            print(f'Processing {filename}')

            # Send the input to the device
            data = data.to(device)
            # Make the predictions
            output = model(data)

            # Get labels
            output = get_label(output, threshhold)

            if target.dim() != 1:
                target= target.to(device)
                target = get_label(target, threshhold)
                accuracy = accuracy_(target, output)
                f1 = f1_(target, output)
                accuracy_scores.append(accuracy)
                f1_scores.append(f1)

            # Save mask
            else:
                if not os.path.exists(pred_path):
                    os.makedirs(pred_path)
                output_path = os.path.join(pred_path, filename)
                save_prediction(output, output_path)
                prediction_filnames.append(output_path)

    # Print a message after processing all images
    print('Prediction completed.')

    # Print a message after processing all images
    if target.dim() != 1:
        avg_accuracy = sum(accuracy_scores).item() / len(accuracy_scores)
        avg_f1 = sum(f1_scores).item() / len(f1_scores)
        print('F1 Score: ', avg_f1)
        print('Accuracy: ', avg_accuracy)
        return avg_f1, avg_accuracy, prediction_filnames
    else:
        print("accuracy and f1 not computed")
        return None, None,prediction_filnames

def create_augmented_dataset(trainingPath,gtPath,imgAugPath,gtAugPath,rotation_angles= [45,135,225]):

    # Creates directories
    for dirname in (imgAugPath, gtAugPath):
        os.makedirs(dirname, exist_ok=True)
    # Load the original dataset
    images = sorted(os.listdir(trainingPath))
    masks = sorted(os.listdir(gtPath))
    #Select the first 10 images
    for i in range(10):
      for angle in rotation_angles:

        # Get image and mask names
        image_name = images[i]
        mask_name = masks[i]

        # Get images paths
        image_path = os.path.join(trainingPath, image_name)
        mask_path = os.path.join(gtPath, mask_name)

        # Open images
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        to_tensor = transforms.ToTensor()

        #apply the transformations
        image_transformed = transforms.functional.affine(to_tensor(image), angle=angle,translate=(0, 0), scale=1.0, shear=0.0)
        mask_transformed = transforms.functional.affine(to_tensor(mask), angle=angle,translate=(0, 0), scale=1.0, shear=0.0)

        # Convert tensors to PIL Images
        image_transformed_PIL = transforms.ToPILImage()(image_transformed)
        mask_transformed_PIL = transforms.ToPILImage()(mask_transformed)

        # Save augmented dataset
        filename_img = f'Image_{i+1:04d}_{angle:03d}.png'
        filename_gd = f'gdImage_{i+1:04d}_{angle:03d}.png'

        image_path_aug = os.path.join(imgAugPath, filename_img)
        mask_path_aug = os.path.join(gtAugPath, filename_gd)

        image_transformed_PIL.save(image_path_aug)
        mask_transformed_PIL.save(mask_path_aug)

