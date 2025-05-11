"""
Evaluation metrics functions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_roc_and_threshold_curves(y_true, y_probs, num_classes, target_names, title_suffix=""):
    """
    Plot ROC curves and threshold-dependency for multiclass classification.
    
    Args:
        y_true (numpy.ndarray): True class labels
        y_probs (numpy.ndarray): Predicted class probabilities
        num_classes (int): Number of classes
        target_names (list): List of class names
        title_suffix (str, optional): Suffix to add to the plot title
    """
    # Binarize the labels
    y_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Create figure with the new layout: ROC curves on the left, threshold curves on right
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC Curves", "TPR/FPR vs Threshold"))
    
    # Add ROC curves for each class
    for i in range(num_classes):
        fig.add_trace(
            go.Scatter(
                x=fpr[i], y=tpr[i],
                name=f"{target_names[i]} (AUC={roc_auc[i]:.2f})",
                mode="lines",
                hovertemplate='False Positive Rate: %{x:.3f}<br>True Positive Rate: %{y:.3f}'
            ),
            row=1, col=1
        )
    
    # Add diagonal line for random classifier
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            name="Random Classifier",
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False,
            hoverinfo="skip"
        ),
        row=1, col=1
    )
    
    # Add threshold curves for 3 selected classes or less if num_classes < 3
    displayed_classes = min(3, num_classes)
    for i in range(displayed_classes):
        # For each selected class, plot TPR and FPR vs threshold
        thresholds_i = thresholds[i]
        
        # Add trace for TPR vs threshold
        fig.add_trace(
            go.Scatter(
                x=thresholds_i,
                y=tpr[i],
                name=f"{target_names[i]} TPR",
                mode="lines",
                line=dict(width=2),
                showlegend=(i == 0),  # Show in legend only for the first class
                hovertemplate='Threshold: %{x:.3f}<br>True Positive Rate: %{y:.3f}'
            ),
            row=1, col=2
        )
        
        # Add trace for FPR vs threshold
        fig.add_trace(
            go.Scatter(
                x=thresholds_i,
                y=fpr[i],
                name=f"{target_names[i]} FPR",
                mode="lines",
                line=dict(width=2, dash="dash"),
                showlegend=(i == 0),  # Show in legend only for the first class
                hovertemplate='Threshold: %{x:.3f}<br>False Positive Rate: %{y:.3f}'
            ),
            row=1, col=2
        )
    
    # Update layout for ROC subplot
    fig.update_xaxes(title_text="False Positive Rate", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", range=[0, 1], row=1, col=1)
    
    # Update layout for Threshold subplot
    fig.update_xaxes(
        title_text="Threshold Value",
        range=[0, 1],
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="Rate",
        range=[0, 1],
        row=1, col=2
    )
    
    # Update overall layout
    fig.update_layout(
        title=f"ROC & Threshold Analysis {title_suffix}",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        width=1000,
        height=450
    )
    
    # Show plot
    fig.show()

def check_model_numerics(model, input_tensor):
    """
    Check for any infinite or nan values during forward pass.
    
    Args:
        model (nn.Module): Model to check
        input_tensor (torch.Tensor): Input tensor
    """
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

def dataset_verification_report(df, label_col='label'):
    """
    Generate a dataset verification report.
    
    Args:
        df (pd.DataFrame): DataFrame with image data
        label_col (str): Column name containing labels
    """
    print("Starting dataset verification...\n")
    print("--- Dataset Verification Report ---\n")

    # Preview of the dataset
    print("Dataset Preview (First 5 Rows):")
    print(df.head())
    print("-" * 50)

    # Dataset summary
    print("Basic Dataset Info:")
    print(df.info())
    print("-" * 50)

    # Check for missing values
    print("Missing Values in Each Column:")
    print(df.isnull().sum())
    print("-" * 50)

    # Unique values per column
    print("Unique Values per Column:")
    print(df.nunique())
    print("-" * 50)

    # Label distribution
    # Expected values
    expected_counts = {
        'neutrophil': 3329,
        'eosinophil': 3117,
        'ig': 2895,
        'platelet': 2348,
        'erythroblast': 1551,
        'monocyte': 1420,
        'basophil': 1218,
        'lymphocyte': 1214
    }

    # Label distribution
    print("Label Distribution:")
    observed_counts = df[label_col].value_counts()
    print(observed_counts)

    # Check for mismatches
    for label, expected_count in expected_counts.items():
        observed_count = observed_counts.get(label, 0)  # Default to 0 if label is missing
        if observed_count != expected_count:
            print(f"Warning: {label} count is {observed_count}, expected {expected_count}.")
    print("-" * 50)

    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    print(f"Number of Duplicate Rows: {duplicate_count}")
    print("-" * 50)

    # Check the dataframe shape
    df_shape = df.shape
    print(f"The dataframe shape: {df_shape}")
    print("-" * 50)

    print("\n --- End of data verification --- \n")

def print_evaluation_results(train_loader, val_loader, test_loader, model, device, num_classes):
    """
    Print evaluation results for a model.
    
    Args:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        model (nn.Module): Model to evaluate
        device (str): Device to run the model on
        num_classes (int): Number of output classes
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Evaluate on training set
    train_true, train_pred = [], []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            train_true.extend(targets.cpu().numpy())
            train_pred.extend(predicted.cpu().numpy())
    
    # Evaluate on validation set
    val_true, val_pred = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_true.extend(targets.cpu().numpy())
            val_pred.extend(predicted.cpu().numpy())
    
    # Evaluate on test set
    test_true, test_pred, test_probs = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            test_true.extend(targets.cpu().numpy())
            test_pred.extend(predicted.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    train_true = np.array(train_true)
    train_pred = np.array(train_pred)
    val_true = np.array(val_true)
    val_pred = np.array(val_pred)
    test_true = np.array(test_true)
    test_pred = np.array(test_pred)
    test_probs = np.array(test_probs)
    
    # Print results
    print("Training Set Classification Report:")
    print(classification_report(train_true, train_pred))
    
    print("\nValidation Set Classification Report:")
    print(classification_report(val_true, val_pred))
    
    print("\nTest Set Classification Report:")
    print(classification_report(test_true, test_pred))
    
    # Compute accuracy
    train_acc = np.mean(train_true == train_pred)
    val_acc = np.mean(val_true == val_pred)
    test_acc = np.mean(test_true == test_pred)
    
    print(f"\nAccuracy:\nTrain: {train_acc:.4f}\nValidation: {val_acc:.4f}\nTest: {test_acc:.4f}")
    
    return {
        'train_true': train_true,
        'train_pred': train_pred,
        'val_true': val_true,
        'val_pred': val_pred,
        'test_true': test_true,
        'test_pred': test_pred,
        'test_probs': test_probs
    }

def print_evaluation_results_dt(X_train, y_train, X_val, y_val, X_test, y_test, model, num_classes):
    """
    Print evaluation results for a decision tree model.
    
    Args:
        X_train, X_val, X_test: Feature arrays
        y_train, y_val, y_test: Target arrays
        model: Trained model
        num_classes (int): Number of output classes
    """
    # Predict on training set
    train_pred = model.predict(X_train)
    
    # Predict on validation set
    val_pred = model.predict(X_val)
    
    # Predict on test set
    test_pred = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    
    # Print results
    print("Training Set Classification Report:")
    print(classification_report(y_train, train_pred))
    
    print("\nValidation Set Classification Report:")
    print(classification_report(y_val, val_pred))
    
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, test_pred))
    
    # Compute accuracy
    train_acc = np.mean(y_train == train_pred)
    val_acc = np.mean(y_val == val_pred)
    test_acc = np.mean(y_test == test_pred)
    
    print(f"\nAccuracy:\nTrain: {train_acc:.4f}\nValidation: {val_acc:.4f}\nTest: {test_acc:.4f}")
    
    return {
        'train_true': y_train,
        'train_pred': train_pred,
        'val_true': y_val,
        'val_pred': val_pred,
        'test_true': y_test,
        'test_pred': test_pred,
        'test_probs': test_probs
    }

def plot_model_performance(model, class_names=None, model_details=None):
    """
    Plot model performance metrics.
    
    Args:
        model (nn.Module): Trained model with metrics history
        class_names (list, optional): List of class names
        model_details (dict, optional): Dictionary with model details
    """
    if not hasattr(model, 'get_all_metrics') or not callable(model.get_all_metrics):
        print("Model does not have a get_all_metrics method")
        return
    
    metrics = model.get_all_metrics()
    if not metrics or not all(key in metrics for key in ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy']):
        print("Model metrics are incomplete or not available")
        return
    
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Create a 2x1 subplot grid (losses and accuracies)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    axs[0].plot(epochs, metrics['train_loss'], 'b-', label='Train')
    axs[0].plot(epochs, metrics['val_loss'], 'r-', label='Validation')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axs[1].plot(epochs, metrics['train_accuracy'], 'b-', label='Train')
    axs[1].plot(epochs, metrics['val_accuracy'], 'r-', label='Validation')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Add model details as text if provided
    if model_details is not None:
        detail_text = (
            f"Model: {model_details.get('model_name', 'N/A')}\n"
            f"Loss Fn: {model_details.get('loss_function', 'N/A')}\n"
            f"Optimizer: {model_details.get('optimizer', 'N/A')}\n"
            f"Learning Rate: {model_details.get('learning_rate', 'N/A')}\n"
            f"Epochs: {model_details.get('epochs', 'N/A')}"
        )
        
        fig.text(
            0.02, 0.97, detail_text,
            ha='left', va='top',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )
    
    plt.tight_layout()
    plt.show() 