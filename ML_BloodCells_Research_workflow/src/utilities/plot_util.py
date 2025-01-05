import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


# palette = ["#F72585", "#7209B7", "#3A0CA3", "#4361EE", "#4CC9F0"]
# color_palette1 = {
#     'train': '#4B0082',
#     'val': '#FF6347',
#     'precision': '#32CD32',
#     'recall': '#1E90FF',
#     'f1': '#FF4500'
# }
# color_palette = {
#     'train': '#4B0082',
#     'val': '#FF6347',
#     'precision': '#F72585',
#     'recall': '#7209B7',
#     'f1': '#3A0CA3'
# }
# def plot_model_performance(model, class_names=None):
def plot_model_performance(model, class_names=None, model_details=None):
    import matplotlib.patches as patches

    color_palette = {
        'train': '#4B0082',
        'val': '#FF6347',
        'precision': '#F72585',
        'recall': '#4361EE',
        'f1': '#3A0CA3'
    }
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold', color='#333333')
    fig.patch.set_facecolor('white')

    axs[0, 0].plot(model.history['train_loss'], label='Train Loss', color=color_palette['train'], linewidth=2, marker=".")
    axs[0, 0].plot(model.history['val_loss'], label='Validation Loss', color=color_palette['val'], linewidth=2, marker=".")
    axs[0, 0].set_title('Loss', fontweight='bold')
    axs[0, 0].set_xlabel('Epochs', color='#555555')
    axs[0, 0].set_ylabel('Loss', color='#555555')
    axs[0, 0].legend()

    axs[0, 1].plot(model.history['train_acc'], label='Train Accuracy', color=color_palette['train'], linewidth=2, marker=".")
    axs[0, 1].plot(model.history['val_acc'], label='Validation Accuracy', color=color_palette['val'], linewidth=2, marker=".")
    axs[0, 1].set_title('Accuracy', fontweight='bold')
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
        print("[---------------]")
        print("!!! metric:", metric)
        print("!!! DEBUG: model.history[metric]:", model.history[metric])
        print("!!! DEBUG: label:", label)
        print("[---------------]")

    axs[1, 0].set_title('Validation Metrics', fontweight='bold')
    axs[1, 0].set_xlabel('Epochs', color='#555555')
    axs[1, 0].set_ylabel('Score', color='#555555')
    axs[1, 0].legend()

    conf_matrix = np.array(model.history['confusion_matrix'][-1])

    # Generate class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in range(conf_matrix.shape[0])]

    sns.heatmap(conf_matrix,
                cmap='RdPu',
                # vmin=1.56,
                # vmax=4.15,
                square=True,
                linewidth=0.3,
                # cbar_kws={'shrink': .72},
                annot_kws={'size': 12},
                annot=True,
                fmt='d',
                ax=axs[1, 1],
                xticklabels=class_names,
                yticklabels=class_names
                # cbar=False)
                )
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

        # Add model details at the top
    if model_details:
        detail_text = (
                r"$\bf{Model\ Name:}$" + f" {model_details.get('model_name', 'N/A')}\n"
                                         r"$\bf{Loss\ Function:}$" + f" {model_details.get('loss_function', 'N/A')}\n"
                                                                     r"$\bf{Optimizer:}$" + f" {model_details.get('optimizer', 'N/A')}\n"
                                                                                            r"$\bf{Accuracy\ Metric:}$" + f" {model_details.get('accuracy_metric', 'N/A')}\n"
                                                                                                                          r"$\bf{Learning\ Rate:}$" + f" {model_details.get('learning_rate', 'N/A')}\n"
                                                                                                                                                      r"$\bf{Epochs:}$" + f" {model_details.get('epochs', 'N/A')}"
        )
        # fig.text(0.5, 0.96, detail_text, ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round'))
        fig.text(
            0.02, 0.98, detail_text,
            ha='left', va='top', fontsize=8,
            color='#333333',
            # color='#555555',
            bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.2', alpha=0.9),
            usetex=False
        )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # plt.tight_layout()
    plt.show()

    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None, model_details=None, figsize=(12, 8)):
    """
    Plots a confusion matrix with model details.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_names (list, optional): List of class names. Defaults to None, which will generate class names as 'Class 0', 'Class 1', etc.
        model_details (dict, optional): Dictionary with model details. Defaults to None.
        figsize (tuple, optional): Size of the figure. Defaults to (12, 8).
    """
    conf_matrix = np.array(confusion_matrix(y_true, y_pred))
    if class_names is None:
        class_names = [f'Class {i}' for i in range(conf_matrix.shape[0])]

    fig, axs = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(conf_matrix,
                cmap='RdPu',
                square=True,
                linewidth=0.3,
                annot_kws={'size': 12},
                annot=True,
                fmt='d',
                ax=axs,
                xticklabels=class_names,
                yticklabels=class_names)

    axs.set_title('Confusion Matrix', fontweight='bold')
    axs.set_xlabel('Model Prediction', color='#555555')
    axs.set_ylabel('True Labels', color='#555555')

    # Add model details
    if model_details is not None:
        detail_text = (
                r"$\bf{Model\ Name:}$" + f" {model_details.get('model_name', 'N/A')}\n"
                                         r"$\bf{Loss\ Function:}$" + f" {model_details.get('loss_function', 'N/A')}\n"
                                                                     r"$\bf{Optimizer:}$" + f" {model_details.get('optimizer', 'N/A')}\n"
                                                                                            r"$\bf{Accuracy\ Metric:}$" + f" {model_details.get('accuracy_metric', 'N/A')}\n"
                                                                                                                          r"$\bf{Learning\ Rate:}$" + f" {model_details.get('learning_rate', 'N/A')}\n"
                                                                                                                                                      r"$\bf{Epochs:}$" + f" {model_details.get('epochs', 'N/A')}"
        )

        fig.text(
            0.02, 0.98, detail_text,
            ha='left', va='top', fontsize=8,
            color='#333333',
            bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.2', alpha=0.9)
        )

    plt.tight_layout()
    plt.show()


# plot the training and validation loss
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss", color="blue", marker="o")
    plt.plot(history["val_loss"], label="Validation Loss", color="red", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


# plot the training and validation accuracy
def plot_accuracy(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history["val_acc"], label="Validation Accuracy", color="green", marker="o")
    plt.plot(history["test_acc"], label="Test Accuracy", color="purple", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation and Test Accuracy")
    plt.legend()
    plt.show()


# plot the training and validation precision
def plot_precision(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history["val_precision"], label="Validation Precision", color="orange", marker="o")
    plt.plot(history["test_precision"], label="Test Precision", color="brown", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Validation and Test Precision")
    plt.legend()
    plt.show()


# plot all metrics
def plot_metrics(model: torch.nn.Module):
    history = model.history
    plot_loss(history)
    plot_accuracy(history)
    plot_precision(history)
