import torch
import numpy as np
import time
from torchmetrics import Accuracy, Precision, Recall, F1Score

def setup_vit_model(config):
    """
    Setup ViT model with appropriate parameters from the config.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        ViTClassifier: Initialized ViT model
    """
    from models.neural_networks import ViTClassifier
    model = ViTClassifier(num_classes=config['num_classes'])
    return model

def setup_dt_model(config):
    """
    Setup a decision tree model.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        DecisionTreeClassifier: Initialized decision tree model
    """
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    )
    return model

def train_dt_model(model, X_train, y_train, X_val, y_val):
    """
    Train a decision tree model and evaluate it.
    
    Args:
        model: Decision tree model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        dict: Training history
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Compute training accuracy
    train_accuracy = model.score(X_train, y_train)
    
    # Compute validation accuracy
    val_accuracy = model.score(X_val, y_val)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    return {'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy}

def evaluate_model_dt(X, y, model):
    """
    Evaluate a decision tree model.
    
    Args:
        X: Features
        y: Labels
        model: Trained model
        
    Returns:
        float: Accuracy score
    """
    return model.score(X, y)

def train_model(epochs, train_loader, val_loader, model, loss_function,
                optimizer, accuracy_metric, device, num_classes,
                early_stopping_patience=None,
                early_stopping_metric='val_loss',
                early_stopping_min_delta=0.001, # Minimal change that would be considered an improvement
                debug=False, print_progress=False):
    """
    Generic model training function with early stopping.
    
    Args:
        epochs (int): Number of epochs to train
        train_loader: Training data loader
        val_loader: Validation data loader
        model: Model to train
        loss_function: Loss function
        optimizer: Optimizer
        accuracy_metric: Accuracy metric function
        device: Device to train on
        num_classes (int): Number of output classes
        early_stopping_patience (int, optional): Number of epochs to wait before stopping
        early_stopping_metric (str): Metric to monitor for early stopping
        early_stopping_min_delta (float): Minimum change to consider as improvement
        debug (bool): Whether to run in debug mode
        print_progress (bool): Whether to print progress
        
    Returns:
        dict: Training history
    """
    # Setup metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Early stopping setup
    if early_stopping_patience:
        best_metric_value = float('inf') if 'loss' in early_stopping_metric else 0
        patience_counter = 0
        best_model_state = None
    
    # Timing
    start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
        # ==== Training phase ====
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        batch_count = 0
        
        for inputs, targets in train_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            
            if debug:
                # Check for NaN values
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Epoch {epoch}, Batch {batch_count}: NaN or Inf loss detected")
                    print("Inputs:", inputs.min().item(), inputs.max().item(), inputs.mean().item())
                    print("Outputs:", outputs.min().item(), outputs.max().item(), outputs.mean().item())
                    break
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Compute accuracy
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == targets).sum().item() / targets.size(0)
            
            # Accumulate metrics
            train_loss += loss.item()
            train_accuracy += accuracy
            batch_count += 1
        
        if batch_count > 0:
            train_loss /= batch_count
            train_accuracy /= batch_count
        
        # ==== Validation phase ====
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                
                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == targets).sum().item() / targets.size(0)
                
                # Accumulate metrics
                val_loss += loss.item()
                val_accuracy += accuracy
                val_batch_count += 1
        
        if val_batch_count > 0:
            val_loss /= val_batch_count
            val_accuracy /= val_batch_count
        
        # Record metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Store metrics in model if supported
        if hasattr(model, 'record_metric'):
            model.record_metric('train_loss', train_loss)
            model.record_metric('train_accuracy', train_accuracy)
            model.record_metric('val_loss', val_loss)
            model.record_metric('val_accuracy', val_accuracy)
        
        # Print progress
        if print_progress:
            time_elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
                  f"Time: {time_elapsed:.2f}s")
        
        # Early stopping check
        if early_stopping_patience:
            # Determine current metric value
            current_metric_value = val_loss if early_stopping_metric == 'val_loss' else \
                                   -val_accuracy if early_stopping_metric == 'val_accuracy' else \
                                   train_loss if early_stopping_metric == 'train_loss' else \
                                   -train_accuracy
            
            # Check if we have an improvement
            is_improvement = False
            
            if 'loss' in early_stopping_metric:  # For loss metrics, lower is better
                is_improvement = current_metric_value < (best_metric_value - early_stopping_min_delta)
            else:  # For accuracy metrics, higher is better (but we're using negative values)
                is_improvement = current_metric_value < (best_metric_value - early_stopping_min_delta)
            
            if is_improvement:
                best_metric_value = current_metric_value
                patience_counter = 0
                # Save best model state
                best_model_state = {key: val.cpu().clone() for key, val in model.state_dict().items()}
            else:
                patience_counter += 1
                if print_progress:
                    print(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    # Restore best model
                    model.load_state_dict(best_model_state)
                    break
    
    # Training complete
    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f}s")
    
    # Return history
    history = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'training_time': total_time
    }
    
    return history

def train_model_without_stop(epochs, train_loader, val_loader, model, loss_function,
                optimizer, accuracy_metric, device, num_classes, debug=False, print_progress=False):
    """
    Generic model training function without early stopping.
    
    Args:
        epochs (int): Number of epochs to train
        train_loader: Training data loader
        val_loader: Validation data loader
        model: Model to train
        loss_function: Loss function
        optimizer: Optimizer
        accuracy_metric: Accuracy metric function
        device: Device to train on
        num_classes (int): Number of output classes
        debug (bool): Whether to run in debug mode
        print_progress (bool): Whether to print progress
        
    Returns:
        dict: Training history
    """
    # Setup metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Timing
    start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
        # ==== Training phase ====
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        batch_count = 0
        
        for inputs, targets in train_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            
            if debug:
                # Check for NaN values
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Epoch {epoch}, Batch {batch_count}: NaN or Inf loss detected")
                    print("Inputs:", inputs.min().item(), inputs.max().item(), inputs.mean().item())
                    print("Outputs:", outputs.min().item(), outputs.max().item(), outputs.mean().item())
                    break
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Compute accuracy
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == targets).sum().item() / targets.size(0)
            
            # Accumulate metrics
            train_loss += loss.item()
            train_accuracy += accuracy
            batch_count += 1
        
        if batch_count > 0:
            train_loss /= batch_count
            train_accuracy /= batch_count
        
        # ==== Validation phase ====
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                
                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == targets).sum().item() / targets.size(0)
                
                # Accumulate metrics
                val_loss += loss.item()
                val_accuracy += accuracy
                val_batch_count += 1
        
        if val_batch_count > 0:
            val_loss /= val_batch_count
            val_accuracy /= val_batch_count
        
        # Record metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Store metrics in model if supported
        if hasattr(model, 'record_metric'):
            model.record_metric('train_loss', train_loss)
            model.record_metric('train_accuracy', train_accuracy)
            model.record_metric('val_loss', val_loss)
            model.record_metric('val_accuracy', val_accuracy)
        
        # Print progress
        if print_progress:
            time_elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
                  f"Time: {time_elapsed:.2f}s")
    
    # Training complete
    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f}s")
    
    # Return history
    history = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'training_time': total_time
    }
    
    return history

def test_model(test_loader, model, loss_function, accuracy_metric, device, num_classes, print_progress=True):
    """
    Test a model on a test dataset.
    
    Args:
        test_loader: Test data loader
        model: Trained model
        loss_function: Loss function
        accuracy_metric: Accuracy metric function
        device: Device to test on
        num_classes (int): Number of output classes
        print_progress (bool): Whether to print progress
        
    Returns:
        dict: Test metrics
    """
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    batch_count = 0
    
    # Setup classification metrics
    precision_metric = Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    recall_metric = Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
    f1_metric = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
    
    # Collect all predictions and targets for metrics
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == targets).sum().item() / targets.size(0)
            
            # Store predictions and targets
            all_predictions.append(predicted)
            all_targets.append(targets)
            
            # Accumulate metrics
            test_loss += loss.item()
            test_accuracy += accuracy
            batch_count += 1
    
    if batch_count > 0:
        test_loss /= batch_count
        test_accuracy /= batch_count
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Compute additional metrics
    precision = precision_metric(all_predictions, all_targets).item()
    recall = recall_metric(all_predictions, all_targets).item()
    f1 = f1_metric(all_predictions, all_targets).item()
    
    # Print results
    if print_progress:
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
    
    # Return metrics
    test_metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1
    }
    
    return test_metrics 