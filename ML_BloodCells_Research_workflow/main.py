from src.models.model1 import ModelV1
from src.models.model_v8 import TrainerModel8
from src.train_util.train import *
from src.utilities.utils_and_imports import *
from src.utilities.data_file import ImageDataset
from src.models.model0 import ModelV0
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    ConfusionMatrix,
)
from src.utilities.plot_util import *
import torch.optim as optim
import shap

RANDOM_SEED = 42
# load files from the data folder
folder_path = "/Users/yehonatankeypur/Developer/Blood Cells Analysis using Machine Learning/bloodcells_dataset"

# load the data
df = load_files_from_folder(folder_path)
print(df.head())

transform = transforms.Compose(
    [
        transforms.Resize((360, 363)),  # Resize all images to 360x363
        transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
    ]
)

# create the dataset
full_dataset = ImageDataset(df, transform=transform)

# Check if 'df' is already defined
if "df" not in locals() and "df" not in globals():
    # Load the data only if 'df' is not already loaded
    df = load_dataframe("main_DataFrame")
else:
    print("DataFrame 'df' is already loaded.\n")

# Sample a smaller subset (e.g., 10% of the full dataset)
df = df.sample(frac=0.01, random_state=RANDOM_SEED)

# Perform stratified splitting
# Split ratio: 70% training, 15% validation, 15% test
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"])

# Create datasets for each split
train_dataset = ImageDataset(train_df, transform=transform)
val_dataset = ImageDataset(val_df, transform=transform)
test_dataset = ImageDataset(test_df, transform=transform)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = train_loader.dataset.classes
print(f"Class names: {class_names}")

# Set device to MPS if available, otherwise default to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Display the current device being used
print(f"Using device: {device}")
if device.type == "mps":
    print("Running on MPS (Metal Performance Shaders)")
else:
    print("Running on CPU")


def generate_shap_plot(model, data_loader, device='cpu', num_samples=10):
    """
    Generate SHAP explanations for the model.

    Parameters:
        model (torch.nn.Module): The PyTorch model to explain.
        data_loader (torch.utils.data.DataLoader): DataLoader for the input data.
        device (str): The device to use ('cpu' or 'cuda').
        num_samples (int): Number of samples to use for SHAP explanation.

    Returns:
        None. Displays SHAP plots.
    """
    # Ensure the model is in evaluation mode and moved to the correct device
    print("[BEGIN] [1] : generate_shap_plot")

    model.eval()
    model.to(device)

    # Get a batch of data from the data loader
    for inputs, _ in data_loader:
        inputs = inputs[:num_samples].to(device)  # Limit to `num_samples`
        break

    # Use GradientExplainer for SHAP
    explainer = shap.GradientExplainer(model, inputs)

    # Compute SHAP values
    shap_values = explainer.shap_values(inputs)

    # Convert data back to numpy for visualization
    inputs_numpy = inputs.cpu().detach().numpy()

    # Visualize the SHAP summary plot
    # shap.summary_plot(shap_values, inputs_numpy)

    # Optional: Use image plot if working with image data
    shap.image_plot(shap_values, inputs_numpy)


def explain_with_shap(model, data_loader, device='cpu', background_size=100, test_size=3):
    """
    Generate SHAP explanations using a background and test dataset.

    Parameters:
        model (torch.nn.Module): The trained PyTorch model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): Device to use ('cpu' or 'cuda').
        background_size (int): Number of background samples for SHAP.
        test_size (int): Number of test samples for SHAP explanation.

    Returns:
        None. Displays SHAP plots.
    """
    # Ensure the model is in evaluation mode and on the correct device
    print("[BEGIN] [2] : explain_with_shap")
    model.eval()
    model.to(device)

    # Retrieve a batch of data from the loader
    batch = next(iter(data_loader))
    images, _ = batch

    # Split into background and test samples
    background = images[:background_size].to(device)
    test_images = images[background_size:background_size + test_size].to(device)

    # Initialize SHAP DeepExplainer
    explainer = shap.DeepExplainer(model, background)

    # Compute SHAP values for test images
    shap_values = explainer.shap_values(test_images)

    # Convert SHAP values and test images to numpy for visualization
    shap_numpy = list(np.transpose(shap_values, (4, 0, 2, 3, 1)))  # Adjust shape for SHAP visualization
    test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)  # Adjust shape for image plot

    # Plot feature attributions using SHAP
    shap.image_plot(shap_numpy, -test_numpy)


from collections import defaultdict


def explain_correct_and_incorrect(model, data_loader, device='cpu', num_samples=5, background_size=100):
    """
    Generate SHAP explanations for correct and incorrect predictions for each class.

    Parameters:
        model (torch.nn.Module): The trained PyTorch model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (str): Device to use ('cpu' or 'cuda').
        num_samples (int): Number of correct and incorrect samples per class.
        background_size (int): Number of background samples for SHAP.

    Returns:
        None. Displays SHAP plots for each class.
    """
    print("[BEGIN] [3] : explain_correct_and_incorrect")
    model.eval()
    model.to(device)

    # Store correct and incorrect samples for each class
    correct_samples = defaultdict(list)
    incorrect_samples = defaultdict(list)

    # Identify correct and incorrect samples
    for images, labels in tqdm(data_loader, desc="Processing Data"):
        images, labels = images.to(device), labels.to(device)
        predictions = torch.argmax(model(images), dim=1)

        for img, label, pred in zip(images, labels, predictions):
            if label == pred:
                if len(correct_samples[label.item()]) < num_samples:
                    correct_samples[label.item()].append(img.cpu())
            else:
                if len(incorrect_samples[label.item()]) < num_samples:
                    incorrect_samples[label.item()].append(img.cpu())

    # Plot SHAP explanations for each class
    for cls in range(model.block[-1].out_features):  # Assumes last layer defines number of classes
        print(f"\nClass {cls}:")
        background = torch.stack([img for sublist in correct_samples.values() for img in sublist[:background_size]])
        background = background[:background_size].to(device)  # Ensure the background size

        # Plot correct samples
        if correct_samples[cls]:
            print(f"  Correct Predictions:")
            correct_images = torch.stack(correct_samples[cls][:num_samples]).to(device)
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(correct_images)
            shap_numpy = list(np.transpose(shap_values, (4, 0, 2, 3, 1)))
            test_numpy = np.swapaxes(np.swapaxes(correct_images.cpu().numpy(), 1, -1), 1, 2)
            shap.image_plot(shap_numpy, -test_numpy)

        # Plot incorrect samples
        if incorrect_samples[cls]:
            print(f"  Incorrect Predictions:")
            incorrect_images = torch.stack(incorrect_samples[cls][:num_samples]).to(device)
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(incorrect_images)
            shap_numpy = list(np.transpose(shap_values, (4, 0, 2, 3, 1)))
            test_numpy = np.swapaxes(np.swapaxes(incorrect_images.cpu().numpy(), 1, -1), 1, 2)
            shap.image_plot(shap_numpy, -test_numpy)


def run_model_v8(train_model=False, test_model=False, print_model=False, plot_results=False, class_names=None, plot_shap=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 3
    model_8 = TrainerModel8(train_loader, val_loader, test_loader, num_classes=8, device=device)

    model_details = {
        "model_name": model_8.model.__class__.__name__,  # Get the model class name
        "learning_rate": [group['lr'] for group in model_8.optimizer.param_groups],  # Get learning rates for different parameter groups
        "loss_function": model_8.loss_fxn.__class__.__name__,  # Get the loss function name
        "optimizer": model_8.optimizer.__class__.__name__,  # Get the optimizer class name
        "accuracy_metric": model_8.accuracy.__class__.__name__,  # Get the accuracy metric name
        "epochs": epochs,
    }

    if print_model:
        print(model_8)
    if train_model:
        model_8.train(epochs=epochs)
    if test_model:
        model_8.test()
    if plot_results:
        plot_model_performance(model_8, class_names, model_details=model_details)
    if plot_shap:
        # generate_shap_plot(model=model_8.model, data_loader=test_loader, device=device, num_samples=10)
        # explain_with_shap(model=model_8.model, data_loader=test_loader, device=device, background_size=100, test_size=3)
        explain_correct_and_incorrect(model=model_8.model, data_loader=test_loader, device=device, num_samples=5, background_size=100)


run_model_v8(train_model=True,
             test_model=True,
             class_names=class_names,
             print_model=False,
             plot_results=True,
             plot_shap=True
             )
