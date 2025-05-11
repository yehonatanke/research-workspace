"""
Feature extraction module.
"""

import numpy as np
import torch
from PIL import Image
import cv2
from skimage.feature import hog
from skimage import exposure
from transformers import ViTFeatureExtractor, ViTModel
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys'):
    """
    Extract HOG features from a single image.
    
    Args:
        image (numpy.ndarray): Input image (grayscale or RGB)
        orientations (int): Number of orientation bins for HOG
        pixels_per_cell (tuple): Size (in pixels) of a cell
        cells_per_block (tuple): Number of cells in each block
        block_norm (str): Block normalization method ('L2-Hys', 'L1', etc.)
        
    Returns:
        numpy.ndarray: HOG feature vector or None if processing fails
    """
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 2:
            image_gray = image
        else:
            raise ValueError("Unsupported image format for HOG")

        features = hog(
            image_gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            visualize=False
        )
        return features
    except Exception as e:
        print(f"Error extracting HOG features: {e}")
        return None

def process_hog_features(df, img_path_col, target_size=(360, 360), hog_params=None):
    """
    Process HOG features for all images in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with image paths
        img_path_col (str): Column name containing image paths
        target_size (tuple): Target size for images
        hog_params (dict, optional): HOG parameters
        
    Returns:
        tuple: (numpy.ndarray, pd.DataFrame) HOG features and filtered DataFrame
    """
    from data.preprocessing import load_preprocess_image
    
    if hog_params is None:
        hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys'
        }

    hog_features_list = []
    print("Extracting HOG features...")
    for img_path in tqdm(df[img_path_col]):
        img = load_preprocess_image(img_path, target_size=target_size, color_mode='gray')
        if img is not None:
            features = extract_hog_features(img, **hog_params)
            hog_features_list.append(features)
        else:
            hog_features_list.append(None)

    # Filter out None values
    valid_indices = [i for i, f in enumerate(hog_features_list) if f is not None]
    df_filtered = df.iloc[valid_indices].copy()
    hog_features = np.array([hog_features_list[i] for i in valid_indices])

    print(f"HOG Features Extracted. Shape: {hog_features.shape}")
    return hog_features, df_filtered

def load_vit_model(model_name="google/vit-base-patch16-224-in21k", device='cpu'):
    """
    Load a pre-trained ViT model and feature extractor.
    
    Args:
        model_name (str): Name of the pre-trained model
        device (str): Device to load the model on
        
    Returns:
        tuple: (ViTFeatureExtractor, ViTModel) or (None, None) if loading fails
    """
    try:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        model = ViTModel.from_pretrained(model_name).to(device)
        model.eval()  # Set to evaluation mode
        return feature_extractor, model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None

def preprocess_image(image_path, feature_extractor):
    """
    Preprocess an image for ViT feature extraction.
    
    Args:
        image_path (str): Path to the image
        feature_extractor (ViTFeatureExtractor): Feature extractor
        
    Returns:
        dict: Processed inputs or None if processing fails
    """
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = feature_extractor(images=img, return_tensors="pt")
        return inputs
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None

def extract_deep_features(image_path, feature_extractor, model, device='cpu'):
    """
    Extract deep features from an image using ViT.
    
    Args:
        image_path (str): Path to the image
        feature_extractor (ViTFeatureExtractor): Feature extractor
        model (ViTModel): ViT model
        device (str): Device to run the model on
        
    Returns:
        numpy.ndarray: Feature vector or None if extraction fails
    """
    inputs = preprocess_image(image_path, feature_extractor)
    if inputs is None:
        return None

    try:
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return features
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

def extract_features_for_dataset(df, feature_extractor, model, image_path_column="imgPath", device='cpu'):
    """
    Extract deep features for all images in a dataset.
    
    Args:
        df (pd.DataFrame): DataFrame with image paths
        feature_extractor (ViTFeatureExtractor): Feature extractor
        model (ViTModel): ViT model
        image_path_column (str): Column name containing image paths
        device (str): Device to run the model on
        
    Returns:
        tuple: (numpy.ndarray, numpy.ndarray, list) Features, labels, and valid indices
    """
    deep_features_list = []
    valid_indices = []

    print("Extracting Deep Learning features...")
    for idx, img_path in enumerate(tqdm(df[image_path_column], desc="Processing images")):
        features = extract_deep_features(img_path, feature_extractor, model, device)
        if features is not None:
            deep_features_list.append(features)
            valid_indices.append(idx)

    if not deep_features_list:
        raise ValueError("No valid features extracted from the dataset.")

    deep_features = np.array(deep_features_list)
    labels = df.iloc[valid_indices]["label"].values

    print(f"Deep Features Extracted. Shape: {deep_features.shape}")
    if len(valid_indices) != len(df):
        print(f"Warning: Processed {len(valid_indices)} out of {len(df)} images successfully.")

    return deep_features, labels, valid_indices

def scale_features(features):
    """
    Standardize features using StandardScaler.
    
    Args:
        features (numpy.ndarray): Feature array
        
    Returns:
        tuple: (numpy.ndarray, StandardScaler) Scaled features and scaler
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler

def extract_features(loader, model, transform, device):
    """
    Extract features from a dataset using a pre-trained model.
    
    Args:
        loader (DataLoader): Data loader
        model (nn.Module): Model to extract features
        transform (callable): Image transformation
        device (str): Device to run the model on
        
    Returns:
        tuple: (features, labels)
    """
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            batch_features = model(images)  # Get features (without classification)
            features.append(batch_features.cpu().numpy())
            labels.append(targets.numpy())
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    return features, labels

def extract_all_features(train_loader, val_loader, test_loader, vit_model, image_transform, device):
    """
    Extract features from all datasets.
    
    Args:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        vit_model (nn.Module): Model to extract features
        image_transform (callable): Image transformation
        device (str): Device to run the model on
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    print("Extracting features from training data...")
    X_train, y_train = extract_features(train_loader, vit_model, image_transform, device)
    
    print("Extracting features from validation data...")
    X_val, y_val = extract_features(val_loader, vit_model, image_transform, device)
    
    print("Extracting features from test data...")
    X_test, y_test = extract_features(test_loader, vit_model, image_transform, device)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def convert_to_tensors(features_labels):
    """
    Convert features and labels to PyTorch tensors.
    
    Args:
        features_labels (tuple): (features, labels) tuple
        
    Returns:
        tuple: (features_tensor, labels_tensor)
    """
    features, labels = features_labels
    return torch.FloatTensor(features), torch.LongTensor(labels)

def scale_features_split(X_train, X_val, X_test):
    """
    Scale features for train, validation, and test sets.
    
    Args:
        X_train (numpy.ndarray): Training features
        X_val (numpy.ndarray): Validation features
        X_test (numpy.ndarray): Test features
        
    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler 