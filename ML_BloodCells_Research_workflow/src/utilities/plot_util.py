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


def visualize_image_pairs(df, x, y, z):
    """
    Visualizes z pairs of images from a DataFrame: one with specified dimensions (x,y) and
    one random image with dimensions (360,363) from the same label.

    Parameters:
    - df: Pandas DataFrame with columns 'imgPath', 'label', 'dimensions'
    - x, y: integers representing the target dimensions (x,y) to search for
    - z: integer, number of image pairs to visualize
    """
    # Filter images with dimensions (x,y)
    target_dims = (x, y)
    non_standard_df = df[df['dimensions'] == target_dims]

    if non_standard_df.empty:
        print(f"No images found with dimensions {target_dims}")
        return

    # Ensure we don't try to visualize more pairs than available
    z = min(z, len(non_standard_df))

    # Randomly select z images from non-standard dimensions
    selected_non_standard = non_standard_df.sample(n=z, random_state=None)

    # Set up the plot with smaller figure size
    fig, axes = plt.subplots(z, 2, figsize=(6, 3*z))
    if z == 1:
        axes = [axes]  # Make it iterable 

    for idx, (row_idx, row) in enumerate(selected_non_standard.iterrows()):
        # Load the non-standard image
        non_std_path = row['imgPath']
        non_std_label = row['label']
        non_std_img = cv2.imread(str(non_std_path))
        if non_std_img is None:
            print(f"Failed to load image: {non_std_path}")
            continue
        non_std_img = cv2.cvtColor(non_std_img, cv2.COLOR_BGR2RGB)

        # Find a random standard image (360,363) with the same label
        standard_df = df[(df['dimensions'] == (360, 363)) & (df['label'] == non_std_label)]
        if standard_df.empty:
            print(f"No standard (360,363) image found for label {non_std_label}")
            continue
        standard_row = standard_df.sample(n=1, random_state=None).iloc[0]
        std_path = standard_row['imgPath']
        std_img = cv2.imread(str(std_path))
        if std_img is None:
            print(f"Failed to load image: {std_path}")
            continue
        std_img = cv2.cvtColor(std_img, cv2.COLOR_BGR2RGB)

        # Plot non-standard image
        axes[idx][0].imshow(non_std_img)
        axes[idx][0].text(
            10, 10,  # Position
            f'Dims: {target_dims}\nLabel: {non_std_label}',
            ha='left', va='top',
            fontsize=6,
            color='blue',
            bbox=dict(facecolor='white', boxstyle='round', alpha=0.5, edgecolor='none')
        )
        axes[idx][0].axis('off')

        # Plot standard image
        axes[idx][1].imshow(std_img)
        axes[idx][1].text(
            10, 10,  # Position
            f'Dims: (360,363)\nLabel: {non_std_label}',
            ha='left', va='top',
            fontsize=6,
            color='blue',
            bbox=dict(facecolor='white',boxstyle='round', alpha=0.5, edgecolor='none')
        )
        axes[idx][1].axis('off')

    plt.tight_layout()
    plt.show()


def compute_image_entropy(img_path):
    """Calculate entropy of a grayscale image."""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.nan
        hist = np.histogram(img, bins=256, range=[0, 256], density=True)[0]
        hist = hist[hist > 0]
        return entropy(hist, base=2)
    except:
        return np.nan

def plot_dimension_comparison(df, target_dims=(360, 363)):
    """
    Compare images with target dimensions vs. others in a DataFrame.

    Parameters:
    - df: Pandas DataFrame with columns 'imgPath', 'label', 'dimensions'
    - target_dims: Tuple of (width, height) to compare against (default: (360, 363))

    Outputs:
    - Saves two plots: entropy histogram and label distribution bar plot
    """
    # Ensure dimensions are tuples for comparison
    df['dimensions'] = df['dimensions'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # Create a column to identify target dimensions
    df['is_target_dims'] = df['dimensions'].apply(lambda x: 'Target (360, 363)' if x == target_dims else 'Other')

    # Compute entropy for each image
    df['entropy'] = df['imgPath'].apply(compute_image_entropy)

    # Drop rows with NaN entropy (failed to load images)
    df = df.dropna(subset=['entropy'])

    # Plot 1: Entropy distribution comparison
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='entropy', hue='is_target_dims', element='step', stat='density', common_norm=False)
    plt.title('Entropy Distribution: Images with (360, 363) vs. Other Dimensions')
    plt.xlabel('Entropy (bits)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

    # Plot 2: Label distribution comparison
    plt.figure(figsize=(12, 6))
    label_counts = df.groupby(['is_target_dims', 'label']).size().unstack(fill_value=0)
    label_counts.plot(kind='bar', stacked=False, width=0.4)
    plt.title('Label Distribution: Images with (360, 363) vs. Other Dimensions')
    plt.xlabel('Dimension Group')
    plt.ylabel('Count')
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.show()

def calculate_brightness(img_gray):
  """Calculates average pixel intensity (Brightness)."""
  return np.mean(img_gray)

def calculate_contrast(img_gray):
  """Calculates standard deviation of pixel intensities (Contrast)."""
  return np.std(img_gray)

def calculate_sharpness(img_gray):
  """Calculates variance of Sobel gradient magnitude (Sharpness)."""
  grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
  grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
  gradient_magnitude = np.hypot(grad_x, grad_y)
  return np.var(gradient_magnitude)

def calculate_entropy(img_gray):
    """Calculates Shannon entropy of the grayscale histogram."""
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist_prob = hist.ravel() / hist.sum()
    # Filter out zero probabilities before calculation
    return scipy_entropy(hist_prob[hist_prob > 0], base=2)

# --- Main Comparison Function (with Plotly plotting) ---

def compare_image_metrics_plotly(df: pd.DataFrame, target_dimension: tuple = (360, 363)):
    """
    Calculates, averages, and plots image metrics (Brightness, Contrast,
    Sharpness, Entropy) using Plotly, comparing images of a target
    dimension against others.

    Args:
        df (pd.DataFrame): DataFrame with 'imgPath', 'label', 'dimensions' columns.
                           'dimensions' should contain tuples or string representations
                           of tuples (e.g., '(width, height)').
        target_dimension (tuple): The specific dimension (width, height) to compare against.

    Returns:
        pd.DataFrame: A DataFrame containing the average metrics for both groups.
                      Returns None if an error occurs or no images are processed.
    """
    metrics_target = {'Brightness': [], 'Contrast': [], 'Sharpness': [], 'Entropy': []}
    metrics_other = {'Brightness': [], 'Contrast': [], 'Sharpness': [], 'Entropy': []}
    processed_count = 0
    error_count = 0

    print(f"Starting metric calculation for {len(df)} images...")
    print(f"Target dimension: {target_dimension}")

    # Ensure 'dimensions' column is usable
    if not pd.api.types.is_list_like(df['dimensions'].iloc[0]):
        try:
            df['dimensions_tuple'] = df['dimensions'].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x)
            print("Converted 'dimensions' column strings to tuples.")
        except Exception as e:
            print(f"Warning: Could not auto-convert 'dimensions' column: {e}. Trying row-wise parsing.")
            df['dimensions_tuple'] = df['dimensions'] # Keep original for row-wise handling
    else:
         df['dimensions_tuple'] = df['dimensions']

    # Iterate efficiently
    for row in df.itertuples(index=False):
        img_path = row.imgPath
        try:
            # --- 1. Dimension Check ---
            current_dim = row.dimensions_tuple
            if isinstance(current_dim, str):
                 try:
                     current_dim = ast.literal_eval(current_dim)
                 except:
                     print(f"Skipping row: Could not parse dimension '{current_dim}' for image {img_path}")
                     error_count += 1
                     continue

            if not isinstance(current_dim, tuple) or len(current_dim) != 2:
                print(f"Skipping row: Invalid dimension format '{current_dim}' for image {img_path}")
                error_count += 1
                continue

            # --- 2. Image Loading ---
            if not os.path.exists(img_path):
              print(f"Skipping row: Image file not found at {img_path}")
              error_count += 1
              continue

            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f"Skipping row: Failed to load image {img_path}")
                error_count += 1
                continue

            # --- 3. Metric Calculation ---
            brightness = calculate_brightness(img_gray)
            contrast = calculate_contrast(img_gray)
            sharpness = calculate_sharpness(img_gray)
            entropy = calculate_entropy(img_gray)

            # --- 4. Grouping ---
            is_target_dim = (current_dim == target_dimension)

            if is_target_dim:
                metrics_target['Brightness'].append(brightness)
                metrics_target['Contrast'].append(contrast)
                metrics_target['Sharpness'].append(sharpness)
                metrics_target['Entropy'].append(entropy)
            else:
                metrics_other['Brightness'].append(brightness)
                metrics_other['Contrast'].append(contrast)
                metrics_other['Sharpness'].append(sharpness)
                metrics_other['Entropy'].append(entropy)

            processed_count += 1
            if processed_count % 100 == 0:
                 print(f"Processed {processed_count}/{len(df)} images...")

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            error_count += 1

    print(f"\nFinished processing. Processed: {processed_count}, Skipped/Errors: {error_count}")

    if processed_count == 0:
        print("No images were successfully processed.")
        if 'dimensions_tuple' in df.columns:
           df.drop(columns=['dimensions_tuple'], inplace=True)
        return None

    # --- 5. Calculate Averages ---
    avg_metrics = {}
    groups = {'Target Dimension': metrics_target, 'Other Dimensions': metrics_other}
    metric_names = ['Brightness', 'Contrast', 'Sharpness', 'Entropy']

    for group_name, metrics_data in groups.items():
        avg_metrics[group_name] = {}
        for metric in metric_names:
            values = metrics_data[metric]
            if values:
                avg_metrics[group_name][metric] = np.mean(values)
            else:
                avg_metrics[group_name][metric] = np.nan
        avg_metrics[group_name]['Count'] = len(metrics_data.get('Brightness', []))

    # Create results DataFrame
    results_df = pd.DataFrame(avg_metrics)

    if results_df.isnull().all().all():
       print("No valid metrics calculated for plotting.")
       if 'dimensions_tuple' in df.columns:
           df.drop(columns=['dimensions_tuple'], inplace=True)
       return results_df

    # --- 6. Plotting  ---
    print("\nAverage Metrics:")
    print(results_df)

    # Prepare data for plotting (exclude 'Count' row)
    plot_df = results_df.drop('Count')
    metrics_to_plot = plot_df.index.tolist() # Should be ['Brightness', 'Contrast', 'Sharpness', 'Entropy']

    fig = go.Figure()

    # Add bar trace for Target Dimension
    target_count = int(results_df.loc['Count', 'Target Dimension'])
    fig.add_trace(go.Bar(
        x=metrics_to_plot,
        y=plot_df['Target Dimension'].fillna(0), # Use fillna(0) for plotting if NaN
        name=f'Target ({target_dimension}) - Count: {target_count}',
        text=plot_df['Target Dimension'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A'), # Text on bars
        textposition='auto' # Position text automatically
    ))

    # Add bar trace for Other Dimensions
    other_count = int(results_df.loc['Count', 'Other Dimensions'])
    fig.add_trace(go.Bar(
        x=metrics_to_plot,
        y=plot_df['Other Dimensions'].fillna(0), # Use fillna(0) for plotting if NaN
        name=f'Other Dimensions - Count: {other_count}',
        text=plot_df['Other Dimensions'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A'), # Text on bars
        textposition='auto' # Position text automatically
    ))

    # Update layout for grouped bars, titles, etc.
    fig.update_layout(
        barmode='group', # Group bars side-by-side
        title_text='Average Image Metrics Comparison by Dimension Group (Plotly)',
        xaxis_title='Metric',
        yaxis_title='Average Value',
        legend_title='Dimension Group',
        #margin=dict(l=20, r=20, t=50, b=20) 
    )

    # Show the interactive plot
    fig.show()

    # Clean up added column
    if 'dimensions_tuple' in df.columns:
       df.drop(columns=['dimensions_tuple'], inplace=True)

    return results_df

# Metric Calculation Functions (Utilities)

# Calculates average pixel intensity (Brightness)
def calculate_brightness(img_gray):
    return np.mean(img_gray)

# Calculates standard deviation of pixel intensities (Contrast)
def calculate_contrast(img_gray):
    return np.std(img_gray)

# Calculates variance of Sobel gradient magnitude (Sharpness)
def calculate_sharpness(img_gray):
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.hypot(grad_x, grad_y)
    return np.var(gradient_magnitude)

# Calculates Shannon entropy of the grayscale histogram
def calculate_entropy(img_gray):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist_prob = hist.ravel() / hist.sum()
    return scipy_entropy(hist_prob[hist_prob > 0], base=2)

# Main Comparison Function
def compare_image_metrics(df: pd.DataFrame, target_dimension: tuple = (360, 363)):
    """
    Calculates, averages, and plots image metrics (Brightness, Contrast,
    Sharpness, Entropy) creating 4 separate subplots, comparing images of a target
    dimension against others.

    Args:
        df (pd.DataFrame): DataFrame with 'imgPath', 'label', 'dimensions' columns.
                           'dimensions' should contain tuples or string representations
                           of tuples (e.g., '(width, height)').
        target_dimension (tuple): The specific dimension (width, height) to compare against.

    Returns:
        pd.DataFrame: A DataFrame containing the average metrics for both groups.
                      Returns None if an error occurs or no images are processed.
    """
    metrics_target = {'Brightness': [], 'Contrast': [], 'Sharpness': [], 'Entropy': []}
    metrics_other = {'Brightness': [], 'Contrast': [], 'Sharpness': [], 'Entropy': []}
    processed_count = 0
    error_count = 0

    print(f"Starting metric calculation for {len(df)} samples...")
    print(f"Target dimension: {target_dimension}")

    # Ensure 'dimensions' column is usable
    if not pd.api.types.is_list_like(df['dimensions'].iloc[0]):
        try:
            df['dimensions_tuple'] = df['dimensions'].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x)
            print("Converted 'dimensions' column strings to tuples.")
        except Exception as e:
            print(f"Warning: Could not auto-convert 'dimensions' column: {e}. Trying row-wise parsing.")
            df['dimensions_tuple'] = df['dimensions']
    else:
        df['dimensions_tuple'] = df['dimensions']

    # for row in tqdm(df.itertuples(index=False), total=len(df), desc="Processing images"):
    for row in df.itertuples(index=False):
        img_path = row.imgPath
        try:
            # Dimension Check
            current_dim = row.dimensions_tuple
            if isinstance(current_dim, str):
                try:
                    current_dim = ast.literal_eval(current_dim)
                except:
                    print(f"Skipping row: Could not parse dimension '{current_dim}' for image {img_path}")
                    error_count += 1
                    continue

            if not isinstance(current_dim, tuple) or len(current_dim) != 2:
                print(f"Skipping row: Invalid dimension format '{current_dim}' for image {img_path}")
                error_count += 1
                continue

            # Load Image
            if not os.path.exists(img_path):
                print(f"Skipping row: Image file not found at {img_path}")
                error_count += 1
                continue

            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f"Skipping row: Failed to load image {img_path}")
                error_count += 1
                continue

            # Metric Calculation 
            brightness = calculate_brightness(img_gray)
            contrast = calculate_contrast(img_gray)
            sharpness = calculate_sharpness(img_gray)
            entropy = calculate_entropy(img_gray)

            # Grouping 
            is_target_dim = (current_dim == target_dimension)

            if is_target_dim:
                metrics_target['Brightness'].append(brightness)
                metrics_target['Contrast'].append(contrast)
                metrics_target['Sharpness'].append(sharpness)
                metrics_target['Entropy'].append(entropy)
            else:
                metrics_other['Brightness'].append(brightness)
                metrics_other['Contrast'].append(contrast)
                metrics_other['Sharpness'].append(sharpness)
                metrics_other['Entropy'].append(entropy)

            processed_count += 1
            # if processed_count % 100 == 0:
            #     print(f"Processed {processed_count}/{len(df)} images...")

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            error_count += 1

    print(f"\nFinished processing. Processed: {processed_count}, Skipped/Errors: {error_count}")

    if processed_count == 0:
        print("No images were successfully processed.")
        if 'dimensions_tuple' in df.columns:
            df.drop(columns=['dimensions_tuple'], inplace=True)
        return None

    # Calculate Averages
    avg_metrics = {}
    groups = {'Target Dimension': metrics_target, 'Other Dimensions': metrics_other}
    metric_names = ['Brightness', 'Contrast', 'Sharpness', 'Entropy']

    for group_name, metrics_data in groups.items():
        avg_metrics[group_name] = {}
        for metric in metric_names:
            values = metrics_data[metric]
            if values: 
                avg_metrics[group_name][metric] = np.round(np.mean(values), 3)
            else:
                avg_metrics[group_name][metric] = np.nan
        avg_metrics[group_name]['Count'] = len(metrics_data.get('Brightness', []))

    # Create results DataFrame
    results_df = pd.DataFrame(avg_metrics)

    if results_df.isnull().all().all():
        print("No valid metrics calculated for plotting.")
        if 'dimensions_tuple' in df.columns:
            df.drop(columns=['dimensions_tuple'], inplace=True)
        return results_df

    # Plotting 
    print("\nAverage Metrics:")
    print(results_df)

    # Prepare data for plotting (exclude 'Count' row)
    plot_df = results_df.drop('Count')
    metrics_to_plot = plot_df.index.tolist()  # ['Brightness', 'Contrast', 'Sharpness', 'Entropy']

    # Create a 2x2 subplot grid
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=metrics_to_plot,  
        vertical_spacing=0.2,
        horizontal_spacing=0.2,
        # row_heights=[0.5, 0.5],  
        # column_widths=[0.5, 0.5] 
    )

    group_names = ['Target', 'Other']
    target_count = int(results_df.loc['Count', 'Target Dimension'])
    other_count = int(results_df.loc['Count', 'Other Dimensions'])

    for idx, metric in enumerate(metrics_to_plot, 1):
        row = (idx - 1) // 2 + 1  
        col = (idx - 1) % 2 + 1   

        # Values for the current metric
        target_value = plot_df.loc[metric, 'Target Dimension']
        other_value = plot_df.loc[metric, 'Other Dimensions']

        # Bar for Target Dimension
        fig.add_trace(
            go.Bar(
                x=[group_names[0]],
                y=[target_value if pd.notna(target_value) else 0],
                name=f'Target ({target_dimension}) - Count: {target_count}' if idx == 1 else '',
                text=[f'{target_value:.2f}' if pd.notna(target_value) else 'N/A'],
                textposition='auto',
                marker_color='#636EFA',
                showlegend=(idx == 1)  # Show legend only for the first subplot
            ),
            row=row,
            col=col
        )

        # Bar for Other Dimensions
        fig.add_trace(
            go.Bar(
                x=[group_names[1]],
                y=[other_value if pd.notna(other_value) else 0],
                name=f'Other Dimensions - Count: {other_count}' if idx == 1 else '',
                text=[f'{other_value:.2f}' if pd.notna(other_value) else 'N/A'],
                textposition='auto',
                marker_color='#EF553B',
                showlegend=(idx == 1)  # Show legend only for the first subplot
            ),
            row=row,
            col=col
        )

    fig.update_layout(
        title_text='Average Image Metrics Comparison by Dimension Group (Plotly)',
        barmode='group',
        showlegend=True,
        height=800,  
        width=1200,  
        legend_title='Dimension Group'
    )

    for i in range(1, 5):
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1
        fig.update_yaxes(title_text='Average Value', row=row, col=col)

    fig.update_layout(
        title_text="Image Quality Comparison Across Dimension Groups ((360,363) vs. Other)",
        height=600,
        width=1000
    )

    print()
    fig.show()

    # Clean up added column
    if 'dimensions_tuple' in df.columns:
        df.drop(columns=['dimensions_tuple'], inplace=True)

    return results_df
