import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights

class SimpleNN(nn.Module):
    """
    Simple neural network with a single hidden layer.
    
    Attributes:
        input_dimension (int): Input feature dimension
        hidden_layer_units (int): Number of hidden units
        output_dimension (int): Number of output classes
        model_architecture (nn.Sequential): Model layers
        metrics (dict): Dictionary to store metrics
    """
    def __init__(self, input_dimension: int, hidden_layer_units: int, output_dimension: int):
        """
        Initialize the model.
        
        Args:
            input_dimension (int): Input feature dimension
            hidden_layer_units (int): Number of hidden units
            output_dimension (int): Number of output classes
        """
        super(SimpleNN, self).__init__()
        
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        self.model_architecture = nn.Sequential(
            # First fully connected layer
            nn.Linear(input_dimension, hidden_layer_units),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second fully connected layer
            nn.Linear(hidden_layer_units, hidden_layer_units // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output layer
            nn.Linear(hidden_layer_units // 2, output_dimension)
        )
        
    def forward(self, input_tensor: torch.Tensor):
        """Forward pass through the network."""
        return self.model_architecture(input_tensor)
    
    def record_metric(self, metric_name: str, value: float):
        """Record a metric value."""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
    
    def get_history(self, metric_name: str):
        """Get history for a specific metric."""
        return self.metrics.get(metric_name, [])
    
    def get_all_metrics(self):
        """Get all metrics."""
        return self.metrics


class LightCNN(nn.Module):
    """
    Light CNN model with basic convolutional layers.
    
    Attributes:
        input_shape (int): Input channels
        hidden_units (int): Number of hidden units
        output_shape (int): Number of output classes
        features (nn.Sequential): Feature extraction layers
        classifier (nn.Sequential): Classification layers
        metrics (dict): Dictionary to store metrics
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        """
        Initialize the model.
        
        Args:
            input_shape (int): Input channels (1 for grayscale, 3 for RGB)
            hidden_units (int): Number of hidden units
            output_shape (int): Number of output classes
        """
        super(LightCNN, self).__init__()
        
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Feature extraction
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            # Second convolutional block
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            # Third convolutional block
            nn.Conv2d(in_channels=hidden_units * 2, out_channels=hidden_units * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2)
        )
        
        # Adaptive pooling to handle varying input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 4 * 7 * 7, hidden_units * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden_units * 4, output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        """Forward pass through the network."""
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x
    
    def record_metric(self, metric_name: str, value: float):
        """Record a metric value."""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
            
        else:
            # Log an error if we try to record a metric that doesn't exist
            print(f"Warning: Metric {metric_name} does not exist in metrics dictionary")
    
    def get_history(self, metric_name: str):
        """Get history for a specific metric."""
        if metric_name in self.metrics:
            return self.metrics[metric_name]
        else:
            print(f"Warning: Metric {metric_name} does not exist in metrics dictionary")
            return []
    
    def get_all_metrics(self):
        """Get all metrics."""
        return self.metrics


class CNNModel(nn.Module):
    """
    CNN model with deeper architecture for blood cell classification.
    
    Attributes:
        input_channels (int): Input channels
        hidden_units (int): Base number of filters
        output_classes (int): Number of output classes
        metrics (dict): Dictionary to store metrics
    """
    def __init__(self, input_channels: int, hidden_units: int, output_classes: int):
        """
        Initialize the model.
        
        Args:
            input_channels (int): Input channels (1 for grayscale, 3 for RGB)
            hidden_units (int): Base number of filters
            output_classes (int): Number of output classes
        """
        super(CNNModel, self).__init__()
        
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # First convolutional block
        self.conv1 = nn.Conv2d(input_channels, hidden_units, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_units)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(hidden_units, hidden_units * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_units * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(hidden_units * 2, hidden_units * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_units * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(hidden_units * 4, hidden_units * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_units * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_units * 8 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, output_classes)
    
    def forward(self, x: torch.Tensor):
        """Forward pass through the network."""
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Third block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Fourth block
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Dropout for regularization
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ModelResNet18(nn.Module):
    """
    ResNet18-based model for blood cell classification.
    
    Attributes:
        model (nn.Module): ResNet18 model
        num_classes (int): Number of output classes
    """
    def __init__(self, num_classes=8):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of output classes
        """
        super(ModelResNet18, self).__init__()
        
        # Load pre-trained ResNet18 model
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        
        # Replace the last fully connected layer to match the number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def get_optimizer(self):
        """Get default optimizer for this model."""
        # Fine-tuning: lower learning rate for pre-trained layers, higher for new layers
        optimizer = torch.optim.Adam([
            {'params': list(self.model.parameters())[:-2], 'lr': 1e-4},
            {'params': list(self.model.parameters())[-2:], 'lr': 1e-3}
        ])
        return optimizer
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


class ViTClassifier(nn.Module):
    """
    Vision Transformer (ViT) based model for blood cell classification.
    
    Attributes:
        vit (nn.Module): ViT model
        classifier (nn.Linear): Classification head
        metrics (dict): Dictionary to store metrics
    """
    def __init__(self, num_classes):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of output classes
        """
        super(ViTClassifier, self).__init__()
        
        # Load pre-trained ViT model
        self.vit = models.vit_b_16(weights='DEFAULT')
        
        # Replace the classification head
        self.vit.heads = nn.Linear(self.vit.hidden_dim, num_classes)
        
        # Metrics dictionary
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.vit(x)
    
    def record_metric(self, metric_name: str, value: float):
        """Record a metric value."""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
    
    def get_history(self, metric_name: str):
        """Get history for a specific metric."""
        return self.metrics.get(metric_name, [])
    
    def get_all_metrics(self):
        """Get all metrics."""
        return self.metrics


class ResNetTrainer(nn.Module):
    """
    Trainer class for ResNet models.
    
    This class wraps the ResNet model and provides training functionality.
    
    Attributes:
        device (str): Device to use ('cpu' or 'cuda')
        model (nn.Module): ResNet model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        num_classes (int): Number of output classes
        criterion (nn.Module): Loss function
        optimizer (Optimizer): Optimizer
        metrics (dict): Dictionary to store metrics
    """
    def __init__(self, train_loader, val_loader, test_loader=None, num_classes=8, device='cpu'):
        """
        Initialize the trainer.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (DataLoader, optional): Test data loader
            num_classes (int): Number of output classes
            device (str): Device to use ('cpu' or 'cuda')
        """
        super(ResNetTrainer, self).__init__()
        
        self.device = device
        self.model = ModelResNet18(num_classes=num_classes).to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.model.get_optimizer()
        
        # Initialize metrics
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def training_step(self, x, y):
        """
        Training step for a batch.
        
        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor
            
        Returns:
            dict: Dictionary with loss and accuracy
        """
        # Move data to device
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        
        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / y.size(0)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item(), 'accuracy': accuracy}
    
    def validation_step(self, x, y):
        """
        Validation step for a batch.
        
        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor
            
        Returns:
            dict: Dictionary with loss and accuracy
        """
        # Move data to device
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y).sum().item()
            accuracy = correct / y.size(0)
        
        return {'loss': loss.item(), 'accuracy': accuracy}
    
    def process_batch(self, loader, step):
        """
        Process a full batch.
        
        Args:
            loader (DataLoader): Data loader
            step (callable): Step function (training_step or validation_step)
            
        Returns:
            tuple: (total_loss, total_accuracy)
        """
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for x, y in loader:
            batch_results = step(x, y)
            total_loss += batch_results['loss']
            total_accuracy += batch_results['accuracy']
            num_batches += 1
        
        return total_loss / num_batches, total_accuracy / num_batches
    
    def train(self, epochs, print_progress=False):
        """
        Train the model.
        
        Args:
            epochs (int): Number of epochs
            print_progress (bool): Whether to print progress
            
        Returns:
            dict: Training metrics
        """
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss, train_accuracy = self.process_batch(self.train_loader, self.training_step)
            
            # Validation phase
            self.model.eval()
            val_loss, val_accuracy = self.process_batch(self.val_loader, self.validation_step)
            
            # Record metrics
            self.metrics['train_loss'].append(train_loss)
            self.metrics['train_accuracy'].append(train_accuracy)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_accuracy'].append(val_accuracy)
            
            if print_progress:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        return self.metrics
    
    def test(self, print_result=True):
        """
        Test the model on the test set.
        
        Args:
            print_result (bool): Whether to print results
            
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        if self.test_loader is None:
            raise ValueError("Test loader is not defined")
        
        self.model.eval()
        test_loss, test_accuracy = self.process_batch(self.test_loader, self.validation_step)
        
        if print_result:
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy 