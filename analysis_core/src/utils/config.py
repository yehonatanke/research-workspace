"""
Configuration settings module.
"""

def get_config():
    """
    Get the default configuration for the blood cell classification system.
    
    Returns:
        dict: Configuration dictionary with model and training parameters
    """
    config = {
        # Dataset parameters
        'data_dir': './dataset',
        'num_classes': 8,
        'batch_size': 32,
        'input_size': (360, 360),
        
        # Model parameters
        'model_type': 'cnn',  # Options: 'simple', 'light_cnn', 'cnn', 'resnet18', 'vit'
        'hidden_units': 128,
        'dropout_rate': 0.4,
        
        # Training parameters
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'early_stopping_patience': 10,
        
        # Feature extraction parameters
        'hog_orientations': 9,
        'hog_pixels_per_cell': (10, 10),
        'hog_cells_per_block': (2, 2),
        'hog_block_norm': 'L2-Hys',
        
        # Image processing
        'target_dimensions': (360, 363),
    }
    
    return config 