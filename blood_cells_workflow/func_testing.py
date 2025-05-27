import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchmetrics import ConfusionMatrix
import torch

device = 'cpu'


def plot_training_metrics(history):
    """
    Plot training metrics (loss, accuracy, precision, recall, F1-score) and the confusion matrix
    in a single figure with 4 subplots.

    Parameters:
    history (dict): A dictionary containing training metrics and confusion matrix.
                    Keys include 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                    'val_precision', 'val_recall', 'val_f1_score', and 'confusion_matrix'.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()

    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()

    # Precision, Recall, F1-score
    axes[1, 0].plot(history['val_precision'], label='Validation Precision')
    axes[1, 0].plot(history['val_recall'], label='Validation Recall')
    axes[1, 0].plot(history['val_f1_score'], label='Validation F1 Score')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Validation Metrics')
    axes[1, 0].legend()

    # Confusion matrix
    conf_matrix = np.array(history['confusion_matrix'][-1])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Predicted Labels')
    axes[1, 1].set_ylabel('True Labels')
    axes[1, 1].set_title('Confusion Matrix')

    plt.tight_layout()
    plt.show()


# Example usage with dummy data:
dummy_history = {
    'train_loss': [0.8, 0.6, 0.4, 0.3],
    'val_loss': [0.9, 0.7, 0.5, 0.4],
    'train_acc': [0.6, 0.7, 0.8, 0.85],
    'val_acc': [0.5, 0.65, 0.75, 0.8],
    'val_precision': [0.4, 0.6, 0.7, 0.75],
    'val_recall': [0.3, 0.5, 0.65, 0.7],
    'val_f1_score': [0.35, 0.55, 0.68, 0.72],
    'confusion_matrix': [
        [[50, 10], [5, 35]],
        [[55, 8], [6, 31]],
        [[60, 5], [4, 31]],
        [[62, 4], [3, 31]]
    ]
}


### plot_training_metrics(dummy_history)
# Dummy input for testing the visualization function
class DummyModel:
    def __init__(self):
        import numpy as np

        # Simulate epochs
        epochs = 50

        # Create dummy history with some realistic training curves
        self.history = {
            # Loss curves with decreasing trend
            'train_loss': np.linspace(1.5, 0.1, epochs) + np.random.normal(0, 0.05, epochs),
            'val_loss': np.linspace(1.5, 0.2, epochs) + np.random.normal(0, 0.1, epochs),

            # Accuracy curves with increasing trend
            'train_acc': np.linspace(0.3, 0.95, epochs) + np.random.normal(0, 0.05, epochs),
            'val_acc': np.linspace(0.3, 0.9, epochs) + np.random.normal(0, 0.1, epochs),

            # Validation metrics with some variation
            'val_precision': np.linspace(0.3, 0.85, epochs) + np.random.normal(0, 0.05, epochs),
            'val_recall': np.linspace(0.3, 0.9, epochs) + np.random.normal(0, 0.05, epochs),
            'val_f1_score': np.linspace(0.3, 0.87, epochs) + np.random.normal(0, 0.05, epochs),

            # Confusion matrix (simulating a binary classification scenario)
            'confusion_matrix': [np.array([
                [80, 20],  # True Negatives, False Positives
                [15, 85]  # False Negatives, True Positives
            ])]
        }


# Example usage
# dummy_model = DummyModel()


# plot_model_performance(dummy_model)
def plot_model_performance(model):
    color_palette = {
        'train': '#4B0082',
        'val': '#FF6347',
        'precision': '#32CD32',
        'recall': '#1E90FF',
        'f1': '#FF4500'
    }
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold', color='#333333')
    fig.patch.set_facecolor('white')

    axs[0, 0].plot(model.history['train_loss'], label='Train Loss', color=color_palette['train'], linewidth=2, marker=".")
    axs[0, 0].plot(model.history['val_loss'], label='Validation Loss', color=color_palette['val'], linewidth=2, marker=".")
    axs[0, 0].set_title('Loss Curves', fontweight='bold')
    axs[0, 0].set_xlabel('Epochs', color='#555555')
    axs[0, 0].set_ylabel('Loss', color='#555555')
    axs[0, 0].legend()

    axs[0, 1].plot(model.history['train_acc'], label='Train Accuracy', color=color_palette['train'], linewidth=2, marker=".")
    axs[0, 1].plot(model.history['val_acc'], label='Validation Accuracy', color=color_palette['val'], linewidth=2, marker=".")
    axs[0, 1].set_title('Accuracy Curves', fontweight='bold')
    axs[0, 1].set_xlabel('Epochs', color='#555555')
    axs[0, 1].set_ylabel('Accuracy', color='#555555')
    axs[0, 1].legend()

    val_metrics = [
        ('val_precision', 'Validation Precision', color_palette['precision']),
        ('val_recall', 'Validation Recall', color_palette['recall']),
        ('val_f1_score', 'Validation F1 Score', color_palette['f1'])
    ]

    for metric, label, color in val_metrics:
        axs[1, 0].plot(model.history[metric], label=label, color=color, linewidth=2, marker=".")

    axs[1, 0].set_title('Validation Metrics', fontweight='bold')
    axs[1, 0].set_xlabel('Epochs', color='#555555')
    axs[1, 0].set_ylabel('Score', color='#555555')
    axs[1, 0].legend()

    conf_matrix = np.array(model.history['confusion_matrix'][-1])

    sns.heatmap(conf_matrix,
                cmap='magma',
                # vmin=1.56,
                # vmax=4.15,
                square=True,
                linewidth=0.3,
                cbar_kws={'shrink': .72},
                annot_kws={'size': 12},
                annot=True,
                fmt='d',
                ax=axs[1, 1],
                cbar=False)
    axs[1, 1].set_title('Confusion Matrix', fontweight='bold')
    axs[1, 1].set_xlabel('Predicted Labels', color='#555555')
    axs[1, 1].set_ylabel('True Labels', color='#555555')

    for ax in axs.flat:
        # for spine in ax.spines.values():
        #     spine.set_visible(False)
        # ax.tick_params(axis='both', length=0)

        ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.tick_params(width=0.5)
        ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)
        ax.tick_params(colors='#555555')
    plt.tight_layout()
    plt.show()

    return fig


# plot_model_performance1(dummy_model)

def confusion_matrix_example():
    # Initialize the ConfusionMatrix metric

    # Initialize the ConfusionMatrix metric
    confusion_matrix_metric = ConfusionMatrix(num_classes=4, task="multiclass").to(device)

    # Simulate first batch of inputs
    predictions = torch.tensor([0, 2, 1, 3]).to(device)
    true_labels = torch.tensor([0, 1, 1, 3]).to(device)

    # Update the metric
    confusion_matrix_metric.update(predictions, true_labels)

    # Compute the confusion matrix
    conf_matrix_1 = confusion_matrix_metric.compute()
    print("First Confusion Matrix:")
    print(conf_matrix_1)
    print("Configurations of ConfusionMatrix:")
    print(f"Number of classes: {confusion_matrix_metric.num_classes}")
    print("\nInternal State of ConfusionMatrix:")
    for key, value in vars(confusion_matrix_metric).items():
        print(f"{key}: {value}")
    # Reset the metric
    confusion_matrix_metric.reset()

    # Simulate second batch of inputs
    new_predictions = torch.tensor([1, 2, 0, 3]).to(device)
    new_true_labels = torch.tensor([1, 2, 0, 2]).to(device)

    # Update with new data
    confusion_matrix_metric.update(new_predictions, new_true_labels)

    # Compute the new confusion matrix
    conf_matrix_2 = confusion_matrix_metric.compute()
    print("Second Confusion Matrix:")
    print(conf_matrix_2)
    print("Configurations of ConfusionMatrix:")
    print(f"Number of classes: {confusion_matrix_metric.num_classes}")
    print("\nInternal State of ConfusionMatrix:")
    for key, value in vars(confusion_matrix_metric).items():
        print(f"{key}: {value}")


# confusion_matrix_example()

