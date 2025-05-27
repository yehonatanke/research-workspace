# Basic libraries
import os
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import random
from tqdm.auto import tqdm
import gc

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Modeling libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchmetrics import Accuracy
from skimage import io
from skimage.restoration import estimate_sigma
from sklearn.model_selection import train_test_split

torch.manual_seed(42)


# sns.set(style='whitegrid', palette='muted')

# Save DataFrame
def save_dataframe(dataframe, file_name):
    try:
        dataframe.to_pickle(f"{file_name}.pkl")
        print(f"DataFrame successfully saved as '{file_name}.pkl'.")
    except Exception as e:
        print(f"An error occurred while saving the DataFrame: {e}")


# Load DataFrame
def load_dataframe(file_name):
    try:
        dataframe = pd.read_pickle(f"{file_name}.pkl")
        print(f"DataFrame successfully loaded from '{file_name}.pkl'.")
        return dataframe
    except Exception as e:
        print(f"An error occurred while loading the DataFrame: {e}")


# Save DataLoader
def save_dataloaders(train_loader, val_loader, test_loader, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Save the datasets (train, val, test)
    torch.save(train_loader.dataset, os.path.join(save_dir, 'train_dataset.pth'))
    torch.save(val_loader.dataset, os.path.join(save_dir, 'val_dataset.pth'))
    torch.save(test_loader.dataset, os.path.join(save_dir, 'test_dataset.pth'))

    # Save the DataLoader configurations
    config = {
        'batch_size': train_loader.batch_size,
        'shuffle': train_loader.shuffle
    }

    torch.save(config, os.path.join(save_dir, 'dataloader_config.pth'))
    print(f"DataLoaders and configurations saved to {save_dir}")


# Load DataLoader
def load_dataloaders(save_dir):
    # Load datasets
    train_dataset = torch.load(os.path.join(save_dir, 'train_dataset.pth'))
    val_dataset = torch.load(os.path.join(save_dir, 'val_dataset.pth'))
    test_dataset = torch.load(os.path.join(save_dir, 'test_dataset.pth'))

    # Load configurations
    config = torch.load(os.path.join(save_dir, 'dataloader_config.pth'))

    # Recreate DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"DataLoaders loaded from {save_dir}")
    return train_loader, val_loader, test_loader


# Processes a directory of folders to create a Pandas DataFrame
def create_image_dataframe(data_folder, output_file):
    """
    Processes a directory of folders to create a Pandas DataFrame.
    Each folder's name is treated as the label for the images inside it.

    Args:
        data_folder (str): Path to the main folder containing subfolders of images.
        output_file (str): Path to save the resulting dataframe as a CSV file.

    Returns:
        pd.DataFrame: A dataframe with columns 'imgPath' and 'label'.
    """
    img_paths = []
    labels = []

    # Traverse each folder in the main data folder
    for label in os.listdir(data_folder):
        label_folder = os.path.join(data_folder, label)
        if os.path.isdir(label_folder):  # Ensure it is a folder
            for img_file in os.listdir(label_folder):
                img_path = os.path.join(label_folder, img_file)
                if os.path.isfile(img_path):  # Ensure it is a file
                    img_paths.append(img_path)
                    labels.append(label)

    df = pd.DataFrame({'imgPath': img_paths, 'label': labels})

    if output_file:
        save_dataframe(df, output_file)

    return df


# Create a Pandas DataFrame from a dataset
def create_filtered_image_dataframe(data_folder, output_file=None, file_extension=".jpg"):
    img_paths = []
    labels = []

    for label in os.listdir(data_folder):
        label_folder = os.path.join(data_folder, label)
        if not os.path.isdir(label_folder):
            continue
        for img_file in os.listdir(label_folder):
            if not img_file.lower().endswith(file_extension):  # Filter by extension
                continue
            img_path = os.path.join(label_folder, img_file)
            img_paths.append(img_path)
            labels.append(label)

    img_paths_series = pd.Series(img_paths, name='imgPath')
    labels_series = pd.Series(labels, name='label')
    df = pd.concat([img_paths_series, labels_series], axis=1)

    if output_file:
        df.to_csv(output_file, index=False)

    return df


# Create a Pandas DataFrame from a dataset
def create_image_label_dataframe(dataset_path):
    # Prepare containers to store image file paths and associated class labels
    image_paths = []
    class_labels = []

    # Iterate over each subdirectory in the dataset directory
    class_directories = os.listdir(dataset_path)
    for class_dir in class_directories:
        class_dir_path = os.path.join(dataset_path, class_dir)  # Full path to the class subdirectory
        if not os.path.isdir(class_dir_path):  # Skip entries that are not directories
            continue
        images = os.listdir(class_dir_path)
        for image_name in images:
            if not image_name.lower().endswith('.jpg'):  # Ensure only JPG files are processed
                continue
            image_path = os.path.join(class_dir_path, image_name)
            image_paths.append(image_path)
            class_labels.append(class_dir)

    # Create a Pandas DataFrame to consolidate image paths and labels
    image_paths_series = pd.Series(image_paths, name='imgPath')
    class_labels_series = pd.Series(class_labels, name='label')
    df = pd.concat([image_paths_series, class_labels_series], axis=1)

    return df


# Prints the memory usage of the current process in MB
def print_memory_usage():
    import psutil

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"RSS (Resident Set Size): {memory_info.rss / 1024 ** 2:.2f} MB")
    print(f"VMS (Virtual Memory Size): {memory_info.vms / 1024 ** 2:.2f} MB")
    print(f"Shared Memory Size: {memory_info.shared / 1024 ** 2:.2f} MB")


# Usage:
# df = load_dataframe('main_dataframe')
# push_notebook_to_github_V5(notebook_name='Copy_of_Blood_Cells_ML-3.ipynb', commit_message='check_01', branch='colab-user')

def setup_colab_environment(dataset_identifier):
    """
    Sets up Kaggle integration and downloads a specified dataset in Google Colab.

    Parameters:
    - dataset_identifier (str): The Kaggle dataset identifier (e.g., "username/dataset-name").

    Usage:
    setup_colab_environment("unclesamulus/blood-cells-image-dataset")
    """


def setup_colab_environment_V0(dataset_identifier):
    """
    Sets up Kaggle integration and downloads a specified dataset in Google Colab.

    Parameters:
    - dataset_identifier (str): The Kaggle dataset identifier (e.g., "username/dataset-name").

    Usage:
    setup_colab_environment("unclesamulus/blood-cells-image-dataset")
    """


def check_model_numerics(model, input_tensor):
    # Check for any infinite or nan values during forward pass
    model.eval()
    with torch.no_grad():
        try:
            # Check input tensor
            print("Input tensor stats:")
            print(f"Input min: {input_tensor.min()}")
            print(f"Input max: {input_tensor.max()}")
            print(f"Input mean: {input_tensor.mean()}")
            print(f"Input std: {input_tensor.std()}")
            print(f"Any nan in input: {torch.isnan(input_tensor).any()}")
            print(f"Any inf in input: {torch.isinf(input_tensor).any()}")

            # Intermediate checks
            x = input_tensor
            for i, layer in enumerate(model.model_architecture):
                x = layer(x)
                print(f"\nLayer {i} ({layer.__class__.__name__}):")
                print(f"Output shape: {x.shape}")
                print(f"Any nan: {torch.isnan(x).any()}")
                print(f"Any inf: {torch.isinf(x).any()}")
                print(f"Min: {x.min()}")
                print(f"Max: {x.max()}")
                print(f"Mean: {x.mean()}")
                print(f"Std: {x.std()}")

                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"NUMERICAL ISSUE DETECTED IN LAYER {i}")
                    break

        except Exception as e:
            print(f"Error during forward pass: {e}")


def load_files_from_folder(folder_path):
    dataset_path = folder_path

    # Prepare containers to store image file paths and associated class labels
    image_paths = []
    class_labels = []

    # Iterate over each subdirectory in the dataset directory
    class_directories = os.listdir(dataset_path)
    for class_dir in class_directories:
        class_dir_path = os.path.join(dataset_path, class_dir)  # Full path to the class subdirectory
        if not os.path.isdir(class_dir_path):  # Skip entries that are not directories
            continue
        images = os.listdir(class_dir_path)
        for image_name in images:
            if not image_name.lower().endswith('.jpg'):  # Ensure only JPG files are processed
                continue
            image_path = os.path.join(class_dir_path, image_name)
            image_paths.append(image_path)
            class_labels.append(class_dir)

    # Create a Pandas DataFrame to consolidate image paths and labels
    image_paths_series = pd.Series(image_paths, name='imgPath')
    class_labels_series = pd.Series(class_labels, name='label')
    df = pd.concat([image_paths_series, class_labels_series], axis=1)
    return df


# --- new functions ---

# Save model results
def save_model_results(model, file_path):
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': model.history
        }, file_path)
        print(f"Model and history successfully saved to '{file_path}'.")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")


# Load model results
def load_model_results(model, file_path):
    try:
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.history = checkpoint['history']
        print(f"Model and history successfully loaded from '{file_path}'.")
        return model.history
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
