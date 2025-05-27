"""
Data preprocessing module.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import cv2
from skimage import exposure

def save_dataframe(dataframe, file_name):
    """
    Save a DataFrame to a pickle file.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to save
        file_name (str): File name (without extension)
    """
    try:
        dataframe.to_pickle(f"{file_name}.pkl")
        print(f"DataFrame successfully saved as '{file_name}.pkl'.")
    except Exception as e:
        print(f"An error occurred while saving the DataFrame: {e}")

def load_dataframe(file_name):
    """
    Load a DataFrame from a pickle file.
    
    Args:
        file_name (str): File name (without extension)
        
    Returns:
        pd.DataFrame: Loaded DataFrame or None if error occurs
    """
    try:
        dataframe = pd.read_pickle(f"{file_name}.pkl")
        print(f"DataFrame successfully loaded from '{file_name}.pkl'.")
        return dataframe
    except Exception as e:
        print(f"An error occurred while loading the DataFrame: {e}")
        return None

def create_image_dataframe(data_folder, output_file=None):
    """
    Create a DataFrame from a directory of image folders.
    
    Args:
        data_folder (str): Path to folder containing image class folders
        output_file (str, optional): Output file name to save DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with image paths and labels
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

def create_filtered_image_dataframe(data_folder, output_file=None, file_extension=".jpg"):
    """
    Create a DataFrame from a directory, filtering by file extension.
    
    Args:
        data_folder (str): Path to folder containing image class folders
        output_file (str, optional): Output file name to save DataFrame
        file_extension (str, optional): File extension to filter by
        
    Returns:
        pd.DataFrame: DataFrame with image paths and labels
    """
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

def create_image_label_dataframe(dataset_path):
    """
    Create a DataFrame from a dataset with labeled image folders.
    
    Args:
        dataset_path (str): Path to dataset folder
        
    Returns:
        pd.DataFrame: DataFrame with image paths and class labels
    """
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

def check_image(file_path):
    """
    Check if an image file is valid.
    
    Args:
        file_path (str): Path to image file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        img = Image.open(file_path)
        img.verify()  # Check for file corruption
        return True
    except Exception as e:
        print(f"Corrupted image: {file_path} - {e}")
        return False

def check_dimensions(file_path):
    """
    Get the dimensions of an image.
    
    Args:
        file_path (str): Path to image file
        
    Returns:
        tuple: (width, height) or None if error
    """
    try:
        img = Image.open(file_path)
        return img.size  # (width, height)
    except:
        return None

def load_and_normalize_image(image_path):
    """
    Load and normalize an image to [0-1] range.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        numpy.ndarray: Normalized image array or None if error
    """
    try:
        image = Image.open(image_path)

        # Convert to NumPy array with type conversion
        image_array = np.array(image, dtype=np.float32)

        # Normalize pixel values between 0 and 1, handling different image types
        normalized_image = image_array / 255.0

        return normalized_image

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def load_preprocess_image(img_path, target_size=(360, 360), color_mode='gray'):
    """
    Load and preprocess an image.
    
    Args:
        img_path (str): Path to image file
        target_size (tuple): Target dimensions (width, height)
        color_mode (str): 'gray' or 'rgb'
        
    Returns:
        numpy.ndarray: Processed image or None if error
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        if color_mode == 'gray':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif color_mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        return img
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

def sample_by_label_safe(df, label_col, n):
    """
    Sample n items from each label category (or all if fewer).
    
    Args:
        df (pd.DataFrame): DataFrame to sample from
        label_col (str): Column name containing labels
        n (int): Number of samples per label
        
    Returns:
        pd.DataFrame: New DataFrame with sampled rows
    """
    return df.groupby(label_col).apply(lambda x: x.sample(min(n, len(x)), random_state=42)).reset_index(drop=True)

def sample_by_label(df, label_col, n):
    """
    Sample exactly n items from each label category.
    
    Args:
        df (pd.DataFrame): DataFrame to sample from
        label_col (str): Column name containing labels
        n (int): Number of samples per label
        
    Returns:
        pd.DataFrame: New DataFrame with sampled rows
    """
    return df.groupby(label_col).apply(lambda x: x.sample(n, random_state=42)).reset_index(drop=True)

def create_dummy_df_from_folder(folder_path, n=None):
    """
    Create a small DataFrame with n samples per class.
    
    Args:
        folder_path (str): Path to dataset folder
        n (int, optional): Number of samples per class, None for all
        
    Returns:
        pd.DataFrame: DataFrame with sampled images
    """
    dataset_path = folder_path

    image_paths = []
    class_labels = []

    class_directories = os.listdir(dataset_path)
    for class_dir in class_directories:
        class_dir_path = os.path.join(dataset_path, class_dir)
        if not os.path.isdir(class_dir_path):
            continue
        images = os.listdir(class_dir_path)
        count = 0
        for image_name in images:
            if not image_name.lower().endswith('.jpg'):
                continue
            if n is not None and count >= n:
                break
            image_path = os.path.join(class_dir_path, image_name)
            image_paths.append(image_path)
            class_labels.append(class_dir)
            count += 1

    # Create a Pandas DataFrame to consolidate image paths and labels
    image_paths_series = pd.Series(image_paths, name='imgPath')
    class_labels_series = pd.Series(class_labels, name='label')
    df = pd.concat([image_paths_series, class_labels_series], axis=1)
    return df

def sample_n_per_label(df, n):
    """
    Create a new DataFrame with n random samples from each label.
    
    Args:
        df (pd.DataFrame): DataFrame to sample from
        n (int): Number of samples per label
        
    Returns:
        pd.DataFrame: New DataFrame with n samples per label
    """
    # Group by label and sample n from each group
    dummy_df = df.groupby('label').apply(lambda x: x.sample(n, random_state=42)).reset_index(drop=True)
    return dummy_df 