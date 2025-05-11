"""
Dataset module.

This module contains PyTorch dataset classes and data loading utilities.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    """
    PyTorch Dataset for blood cell images.
    
    Attributes:
        dataframe (pd.DataFrame): DataFrame with image paths and labels
        transform (callable, optional): Optional transform to be applied to images
        classes (list): List of unique class labels
        class_to_idx (dict): Mapping from class names to indices
        idx_to_class (dict): Mapping from indices to class names
    """
    def __init__(self, dataframe, transform=None):
        """
        Initialize the dataset.
        
        Args:
            dataframe (pd.DataFrame): DataFrame with image paths and labels
            transform (callable, optional): Optional transform to be applied to images
        """
        self.dataframe = dataframe
        self.transform = transform
        self.classes = sorted(self.dataframe['label'].unique())
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label) where label is the class index
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx]['imgPath']
        label_str = self.dataframe.iloc[idx]['label']
        label = self.class_to_idx[label_str]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def save_dataloaders(train_loader, val_loader, test_loader, save_dir):
    """
    Save DataLoader objects to disk.
    
    Args:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        save_dir (str): Directory to save the data loaders
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_loader.dataset, os.path.join(save_dir, 'train_dataset.pth'))
    torch.save(val_loader.dataset, os.path.join(save_dir, 'val_dataset.pth'))
    torch.save(test_loader.dataset, os.path.join(save_dir, 'test_dataset.pth'))

    config = {
        'batch_size': train_loader.batch_size,
        'shuffle': train_loader.shuffle
    }

    torch.save(config, os.path.join(save_dir, 'dataloader_config.pth'))
    print(f"DataLoaders and configurations saved to {save_dir}")

def load_dataloaders(save_dir):
    """
    Load DataLoader objects from disk.
    
    Args:
        save_dir (str): Directory containing saved data loaders
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = torch.load(os.path.join(save_dir, 'train_dataset.pth'))
    val_dataset = torch.load(os.path.join(save_dir, 'val_dataset.pth'))
    test_dataset = torch.load(os.path.join(save_dir, 'test_dataset.pth'))
    config = torch.load(os.path.join(save_dir, 'dataloader_config.pth'))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"DataLoaders loaded from {save_dir}")
    return train_loader, val_loader, test_loader

def check_split_proportions(original_df, *splits):
    """
    Check if the splitting of data was done properly.
    
    Args:
        original_df (pd.DataFrame): Original DataFrame
        *splits: Split DataFrames to check
    """
    print("=== Split Proportions ===")
    print(f"Original dataset size: {len(original_df)}")
    total_split = 0
    
    for i, df in enumerate(splits):
        print(f"Split {i+1} size: {len(df)} ({len(df)/len(original_df)*100:.2f}%)")
        total_split += len(df)
    
    print(f"Total in splits: {total_split} ({total_split/len(original_df)*100:.2f}%)")
    if total_split != len(original_df):
        print(f"WARNING: The splits don't add up to the original dataset size. Difference: {len(original_df) - total_split}")
    print("========================")

def get_class_distribution(dataset):
    """
    Get the distribution of classes in a dataset.
    
    Args:
        dataset (Dataset): PyTorch dataset
        
    Returns:
        dict: Dictionary with class counts
    """
    counts = {}
    for _, label in dataset:
        label_name = dataset.idx_to_class[label] if hasattr(dataset, 'idx_to_class') else label
        counts[label_name] = counts.get(label_name, 0) + 1
    return counts

def create_lr_dataloaders(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, batch_size):
    """
    Create DataLoader objects for logistic regression from numpy arrays.
    
    Args:
        X_train_scaled, X_val_scaled, X_test_scaled: Feature arrays
        y_train, y_val, y_test: Target arrays
        batch_size (int): Batch size
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_vit_transform(weights, vit_input_size):
    """
    Get transformations for ViT model.
    
    Args:
        weights: Model weights with preprocessing transformations
        vit_input_size (tuple): Required input size for ViT
        
    Returns:
        transforms.Compose: Composition of transformations
    """
    return transforms.Compose([
        transforms.Resize(vit_input_size),
        transforms.ToTensor(),
        weights.transforms(),
    ]) 