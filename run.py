import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from ImageDataset import *
from unet import *
from helper import *

# Constants
TEST_IMAGES = 'data/test_set_images/'
PREDICTIONS_PATH= 'predictions/'
FOREGROUND_TRESHOLD = 0.25

DEFAULT_MODEL_PATH_NORMAL = 'models/modelNormal.pth'
DEFAULT_MODEL_PATH_AUG = 'models/modelAug.pth'
SUBMISSION_FILENAME = 'submission.csv'




def main(args: argparse.Namespace) -> None:
    """Main to predict.

    Args:
        args (argparse.Namespace): namespace of arguments.
    """
    print('== Start predicting submission ==')

    # Set model path based on augmentation flag
    model_path = DEFAULT_MODEL_PATH_AUG if not args.aug else DEFAULT_MODEL_PATH_NORMAL
    
    print('Model path:', model_path)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    pin_memory = device == 'cuda'

    # Define transforms
    image_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Define dataset
    test_set = ImagesDataset(
        img_dir=TEST_IMAGES,
        image_transform=image_transform,
    )
    # Define dataloader
    test_loader = DataLoader(
        dataset=test_set,
        num_workers=2,
        pin_memory=pin_memory,
    )

    # Define model
    model = UNet()
    print('Model: UNet')
    model.to(device)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Do prediction
    _,_,filenames= predict(model, test_loader, device, PREDICTIONS_PATH)

    # Create CSV for submission
    masks_to_submission(SUBMISSION_FILENAME, *filenames)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predicting model for road segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--aug',
        choices=('no'),
        help='Use augmented model (modelAug.pth) if yes, else use normal model (modelNormal.pth).',
    )
    args = parser.parse_args()
    main(args)
