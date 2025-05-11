"""
Exploratory data analysis visualization module.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import cv2
from PIL import Image
from skimage import exposure
from skimage.feature import hog

def visualize_dimensions(df, dimensions_column):
    """
    Visualize the distribution of image dimensions.
    
    Args:
        df (pd.DataFrame): DataFrame with dimensions column
        dimensions_column (str): Column name containing dimensions
    """
    # Prepare data
    dimension_counts = df[dimensions_column].value_counts().reset_index()
    dimension_counts.columns = ['Dimensions', 'Count']
    dimension_counts['Dimensions'] = dimension_counts['Dimensions'].astype(str)  # Convert dimensions to string for better labeling
    total_count = dimension_counts['Count'].sum()

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        specs=[[{"type": "xy"}, {"type": "domain"}]]
    )

    fig.add_trace(
        go.Bar(
            x=dimension_counts['Dimensions'],
            y=dimension_counts['Count'],
            text=dimension_counts['Count'],
            textposition='outside',
            hovertemplate=(
                "Image dimensions: %{x}<br>"
                "Image count: %{y}<br>"
                "Percentage from total: %{customdata:.2f}%<extra></extra>"
            ),
            customdata=(dimension_counts['Count'] / total_count * 100),
            showlegend=False
        ),
        row=1, col=1
    )

    # Pie Chart
    fig.add_trace(
        go.Pie(
            labels=dimension_counts['Dimensions'],
            values=dimension_counts['Count'],
            hovertemplate=(
                "Image dimensions: %{label}<br>"
                "Image count: %{value}<br>"
                "Percentage from total: %{percent}<extra></extra>"
            ),
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text="Distribution of Image Dimensions",
        height=600,
        width=1000
    )

    fig.show()

def plot_label_distribution(df, label_column):
    """
    Plot the distribution of labels in the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame with labels
        label_column (str): Column name containing labels
    """
    # Validate input
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in DataFrame.")
    if df.empty:
        raise ValueError("DataFrame is empty.")

    # Compute label distribution and percentages
    total_images = len(df)
    label_counts = df[label_column].value_counts().reset_index()
    label_counts.columns = [label_column, 'count']
    label_counts['percentage'] = (label_counts['count'] / total_images * 100).round(2)

    fig = px.bar(
        label_counts,
        x=label_column,
        y='count',
        title="Proportional Distribution of Labels in the Dataset",
        labels={label_column: 'Label', 'count': 'Count'},
        text='percentage',
        text_auto='.2s',
        width=10,
    )

    fig.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside',
        hovertemplate=(
            '<b>Label:</b> %{x}<br>'
            '<b>Count:</b> %{y}<br>'
            '<b>Percentage of total:</b> %{text}%'
            '<extra></extra>'
        )
    )

    fig.update_layout(
        title_font_size=18,
        xaxis_title='Label',
        yaxis_title='Count',
        showlegend=False,
        width=1100,
        height=520
    )

    fig.show()

def plot_label_distribution_polar(df, label_column):
    """
    Plot polar visualization of label distributions.
    
    Args:
        df (pd.DataFrame): DataFrame with labels
        label_column (str): Column name containing labels
    """
    # Validate input
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in DataFrame.")
    if df.empty:
        raise ValueError("DataFrame is empty.")

    # Calculate the distribution of labels and percentages
    label_counts = df[label_column].value_counts().reset_index()
    label_counts.columns = ['label', 'count']
    total_images = len(df)
    label_counts['percentage'] = (label_counts['count'] / total_images * 100).round(2)

    # Bin the percentages into discrete intervals
    num_bins = 30
    label_counts['percentage_bin'] = pd.cut(
        label_counts['percentage'],
        bins=np.linspace(0, label_counts['percentage'].max(), num_bins + 1),
        labels=[f"{int(i.left)}-{int(i.right)}%" for i in pd.interval_range(0, label_counts['percentage'].max(), num_bins)],
        include_lowest=True
    )

    fig = px.bar_polar(
        label_counts,
        r='percentage',
        theta='label',
        color='percentage_bin',
        labels={'label': 'Label',
                'count': 'Count',
                'percentage': 'Percentage of total',
                'percentage_bin': 'Percentage'},
        title="Polar Visualization of Label Distributions with Proportional Metrics",
        color_discrete_sequence=px.colors.sequential.Plasma_r,
    )

    fig.update_traces(
        hovertemplate=(
            '<b>Label:</b> %{theta}<br>' +
            '<b>Count:</b> %{customdata}<br>' +
            '<b>Percentage of total:</b> %{r:.2f}%<br>'
        ),
        customdata=label_counts[['count']].to_numpy(),
    )

    fig.update_layout(
        polar_radialaxis=dict(showticklabels=False),
        font_size=16,
        legend_font_size=16,
        polar_radialaxis_ticksuffix='%',
        polar_angularaxis_rotation=90,
        width=1100,
        height=520
    )

    fig.show()

def visualize_image_pairs(df, x, y, z):
    """
    Visualizes z pairs of images: one with specified dimensions (x,y) and
    one random image with dimensions (360,363) from the same label.
    
    Args:
        df (pd.DataFrame): DataFrame with image paths, labels, and dimensions
        x (int): Width of target dimension
        y (int): Height of target dimension
        z (int): Number of image pairs to visualize
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

def plot_first_images(df, paths='imgPath', label_col='label', num_images_per_label=4):
    """
    Plot first few images from each label.
    
    Args:
        df (pd.DataFrame): DataFrame with image paths and labels
        paths (str): Column name containing image paths
        label_col (str): Column name containing labels
        num_images_per_label (int): Number of images to show per label
    """
    # Get unique labels
    unique_labels = df[label_col].unique()

    num_labels = len(unique_labels)
    fig, axes = plt.subplots(num_labels, num_images_per_label,
                             figsize=(2*num_images_per_label, 2*num_labels))

    if num_labels == 1:
        axes = axes.reshape(1, -1)

    for i, label in enumerate(unique_labels):
        label_images = df[df[label_col] == label]

        for j in range(min(num_images_per_label, len(label_images))):
            img_path = label_images.iloc[j][paths]
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')

            if j == 0:
                axes[i, j].set_title(label, loc='left', color='blue', fontsize=12)

    plt.tight_layout()
    plt.show()

def visualize_hog_samples(df, img_path_col, label_col, num_samples=3, target_size=(360, 360), hog_params=None):
    """
    Visualize HOG features for a set of sample images.
    
    Args:
        df (pd.DataFrame): DataFrame with image paths and labels
        img_path_col (str): Column name containing image paths
        label_col (str): Column name containing labels
        num_samples (int): Number of sample images to visualize
        target_size (tuple): Target size for images
        hog_params (dict, optional): HOG parameters
    """
    from data.preprocessing import load_preprocess_image
    
    if hog_params is None:
        hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys'
        }

    sample_indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)
    sample_paths = df[img_path_col].iloc[sample_indices]
    sample_labels = df[label_col].iloc[sample_indices]

    print("\nVisualizing HOG Features on Sample Images...")
    plt.figure(figsize=(12, 4 * len(sample_indices)))

    for i, (img_path, label) in enumerate(zip(sample_paths, sample_labels)):
        img_color = load_preprocess_image(img_path, target_size=target_size, color_mode='rgb')
        if img_color is None:
            print(f"Failed to load image: {img_path}")
            continue

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)

        try:
            _, hog_image = hog(
                img_gray,
                visualize=True,
                **hog_params
            )
        except Exception as e:
            print(f"Error computing HOG for {img_path}: {e}")
            continue

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        plt.subplot(len(sample_indices), 3, i * 3 + 1)
        plt.imshow(img_color)
        plt.text(10, 10, f'Original Image\nLabel: {label}', ha='left', va='top',fontsize=9,color='blue',bbox=dict(facecolor='white',boxstyle='round', alpha=0.5, edgecolor='none'))
        plt.axis('off')

        plt.subplot(len(sample_indices), 3, i * 3 + 2)
        plt.imshow(img_gray, cmap='gray')
        plt.text(10, 10, 'Grayscale Image', ha='left', va='top',fontsize=9,color='blue',bbox=dict(facecolor='white',boxstyle='round', alpha=0.5, edgecolor='none'))
        plt.axis('off')

        plt.subplot(len(sample_indices), 3, i * 3 + 3)
        plt.imshow(hog_image_rescaled, cmap='gray')
        plt.text(10, 10, 'HOG Visualization', ha='left', va='top',fontsize=9,color='blue',bbox=dict(facecolor='white',boxstyle='round', alpha=0.5, edgecolor='none'))
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_tsne(X_test_np, y_test_np, num_classes, n_samples_tsne=None):
    """
    Plot t-SNE visualization of feature embeddings.
    
    Args:
        X_test_np (numpy.ndarray): Feature array
        y_test_np (numpy.ndarray): Label array
        num_classes (int): Number of classes
        n_samples_tsne (int, optional): Number of samples to use for t-SNE
    """
    from sklearn.manifold import TSNE
    
    # Sample data if needed
    if n_samples_tsne is not None and n_samples_tsne < len(X_test_np):
        indices = np.random.choice(len(X_test_np), n_samples_tsne, replace=False)
        X_subset = X_test_np[indices]
        y_subset = y_test_np[indices]
    else:
        X_subset = X_test_np
        y_subset = y_test_np
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_subset)-1))
    X_tsne = tsne.fit_transform(X_subset)
    
    # Get unique labels
    unique_labels = np.unique(y_subset)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.title('t-SNE visualization of features')
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    
    for i, label in enumerate(unique_labels):
        indices = y_subset == label
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], c=[colors[label]], label=f'Class {label}')
    
    plt.legend()
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.tight_layout()
    plt.show()

def compare_datasets_images(dataframe1, dataset2, start_idx=0, num_images=8):
    """
    Compare images from a DataFrame and a PyTorch Dataset.
    
    Args:
        dataframe1 (pd.DataFrame): DataFrame with image paths
        dataset2 (Dataset): PyTorch Dataset
        start_idx (int): Starting index
        num_images (int): Number of images to compare
    """
    plt.figure(figsize=(num_images * 2, 4))

    for i in range(num_images):
        idx = start_idx + i

        img_path1 = dataframe1.iloc[idx]['imgPath']
        image1 = Image.open(img_path1)

        image2, label2 = dataset2[idx]
        image2 = image2.permute(1, 2, 0).numpy()
        ax1 = plt.subplot(2, num_images, i + 1)
        ax1.imshow(image1)
        ax1.axis('off')
        ax1.text(0.98, 0.98, f"Range: [0-255] , (Index: {idx})", fontsize=6, color='blue', ha='right', va='top', transform=ax1.transAxes)

        ax2 = plt.subplot(2, num_images, num_images + i + 1)
        ax2.imshow(image2)
        ax2.axis('off')
        ax2.text(0.98, 0.98, f"Range: [0-1] , (Index: {idx})", fontsize=6, color='blue', ha='right', va='top', transform=ax2.transAxes)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def visualize_dataset_splits(train_df, val_df, test_df, label_column='label'):
    """
    Visualize the distribution of labels in dataset splits.
    
    Args:
        train_df (pd.DataFrame): Training DataFrame
        val_df (pd.DataFrame): Validation DataFrame
        test_df (pd.DataFrame): Test DataFrame
        label_column (str): Column name containing labels
    """
    # Validate inputs
    for df, name in zip([train_df, val_df, test_df], ['train_df', 'val_df', 'test_df']):
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in {name}")
        if df.empty:
            raise ValueError(f"{name} is empty")
    
    # Prepare data
    train_counts = train_df[label_column].value_counts().reset_index()
    train_counts.columns = [label_column, 'train_count']
    
    val_counts = val_df[label_column].value_counts().reset_index()
    val_counts.columns = [label_column, 'val_count']
    
    test_counts = test_df[label_column].value_counts().reset_index()
    test_counts.columns = [label_column, 'test_count']
    
    # Merge dataframes
    counts = pd.merge(train_counts, val_counts, on=label_column)
    counts = pd.merge(counts, test_counts, on=label_column)
    
    # Calculate percentages
    train_total = train_df.shape[0]
    val_total = val_df.shape[0]
    test_total = test_df.shape[0]
    
    counts['train_pct'] = (counts['train_count'] / train_total * 100).round(1)
    counts['val_pct'] = (counts['val_count'] / val_total * 100).round(1)
    counts['test_pct'] = (counts['test_count'] / test_total * 100).round(1)
    
    # Prepare for plotting
    counts_melted = pd.melt(
        counts, 
        id_vars=[label_column], 
        value_vars=['train_count', 'val_count', 'test_count'],
        var_name='Split', 
        value_name='Count'
    )
    
    pct_melted = pd.melt(
        counts, 
        id_vars=[label_column], 
        value_vars=['train_pct', 'val_pct', 'test_pct'],
        var_name='Split_pct', 
        value_name='Percentage'
    )
    
    counts_melted['Split'] = counts_melted['Split'].map({
        'train_count': 'Train', 
        'val_count': 'Validation', 
        'test_count': 'Test'
    })
    
    counts_melted['Percentage'] = pct_melted['Percentage']
    
    # Create visualization
    fig = px.bar(
        counts_melted,
        x=label_column,
        y='Count',
        color='Split',
        barmode='group',
        text='Percentage',
        title='Label Distribution Across Train, Validation, and Test Sets',
        labels={label_column: 'Label', 'Count': 'Number of Images'},
        color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96']
    )
    
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        hovertemplate=(
            '<b>Label:</b> %{x}<br>'
            '<b>Count:</b> %{y}<br>'
            '<b>Percentage:</b> %{text}%'
            '<extra></extra>'
        )
    )
    
    fig.update_layout(
        xaxis_title='Label',
        yaxis_title='Count',
        legend_title='Dataset Split',
        width=1100,
        height=600
    )
    
    fig.show()
    
    # Print summary statistics
    print("\nDataset Split Summary:")
    print(f"Train: {train_total} images ({train_total/(train_total+val_total+test_total)*100:.1f}%)")
    print(f"Validation: {val_total} images ({val_total/(train_total+val_total+test_total)*100:.1f}%)")
    print(f"Test: {test_total} images ({test_total/(train_total+val_total+test_total)*100:.1f}%)")

def plot_confusion_matrix(y_true, y_pred, class_names=None, model_details=None, figsize=(7, 5)):
    """
    Plot confusion matrix with model details.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (list, optional): List of class names
        model_details (dict, optional): Dictionary with model details
        figsize (tuple): Figure size
    """
    from sklearn.metrics import confusion_matrix
    
    conf_matrix = np.array(confusion_matrix(y_true, y_pred))
    if class_names is None:
        class_names = [f'Class {i}' for i in range(conf_matrix.shape[0])]

    fig, axs = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(conf_matrix,
                cmap='RdPu',
                square=True,
                linewidth=0.3,
                annot_kws={'size': 8},  # Reduced annotation font size
                annot=True,
                fmt='d',
                ax=axs,
                xticklabels=class_names,
                yticklabels=class_names)

    axs.set_title('Confusion Matrix', fontsize=8)  # Reduced title font size
    axs.set_xlabel('Model Prediction', color='#555555', fontsize=6)  # Reduced x-axis label font size
    axs.set_ylabel('True Labels', color='#555555', fontsize=6)  # Reduced y-axis label font size
    axs.tick_params(axis='x', labelsize=8)
    axs.tick_params(axis='y',color='#555555', labelsize=8)
    
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
            ha='left', va='top', fontsize=6,  # Reduced model details font size
            color='#333333',
            bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.2', alpha=0.9)
        )

    plt.tight_layout()
    plt.show() 