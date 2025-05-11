"""
Entry point for the classification system.

Coordinates all components and workflows.
"""

import os
import torch
import numpy as np
import random
from pathlib import Path

# Local imports
from utils.setup import install, setup_colab_environment, print_config
from utils.config import get_config
from data.preprocessing import create_image_label_dataframe, check_image, check_dimensions
from data.dataset import ImageDataset
from visualization.explorer import plot_label_distribution, visualize_dimensions
from models.training import train_model, test_model
from models.neural_networks import CNNModel, ViTClassifier, SimpleNN, LightCNN

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
def main():
    """Main execution function"""
    # Load configuration
    config = get_config()
    print_config(config)
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup environment and dataset path (if in Colab)
    dataset_path = "unclesamulus/blood-cells-image-dataset"
    
    # Create or load dataframe
    df = create_image_label_dataframe(dataset_path)
    
    # Validate images
    df['valid_image'] = df['imgPath'].apply(check_image)
    df['dimensions'] = df['imgPath'].apply(check_dimensions)
    
    # Data visualization
    plot_label_distribution(df, 'label')
    visualize_dimensions(df, 'dimensions')
    
    # Example of model creation and training (simplified)
    # In a real scenario, you would implement a complete pipeline
    
    print("Blood Cell Classification System initialized.")
    
if __name__ == "__main__":
    main() 