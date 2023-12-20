# Machine Learning: Project 2 - Road Segmentation

The goal of this project is to segment satellite images by detecting roads.
Our classifier consists of a convolutional neural network called UNet.

## Team members

- Ahmad Bilal KAKAR
- Imane Zaaraoui
- Lina Bousbina

## Installation

To run the code of this project, you need to install the libraries listed in
the `requirements.txt` file. You can perform the installation using this
command:
```
pip install -r requirements.txt
```

Dependencies:
- matplotlib
- numpy
- pillow
- scikit-image
- torch
- torchvision
- tqdm


## Predictions for AIcrowd

To reproduce our submission on
[AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation), run:
```
python run.py
```
This command will create the predicted mask for each test image in the
`predictions` directory. It will also produce the `submission.csv` file for submission.

## Structure

This is the structure of the repository:

- `data`: contains the datasets
- `notebooks`: contains the notebooks
    - `normal_training.ipynb`: if you want to train the model on the original training set, run this notebook until the cell where we save the model. It also performs predictions, saves predicted masks, creates submission files, and applies post-processing
    - `augmented_data_training`: similar to `normal_training.ipynb`, but with data augmentation during training.
    - `cross_validation.ipynb`: performs cross-validation to determine the optimal split ratio for training and validation.
    
- `models`: contains the trained models
- `run.py`: script for making predictions for AIcrowd
- `helper.py`: contains the helper functions
- `ImageDataset.py`: dataset class
- `loss.py`: DiceLoss class
- `unet.py`: UNet model

