def plot_distributions(train_dist, test_dist):
    plt.style.use('ggplot')
    labels = sorted(train_dist.keys())
    train_counts = [train_dist[label] for label in labels]
    test_counts = [test_dist[label] for label in labels]

    plt.figure(figsize=(12, 7), dpi=100)

    bar_width = 0.35
    color1 = '#34495e'
    color2 = '#e74c3c'

    r1 = np.arange(len(labels))
    r2 = [x + bar_width for x in r1]

    plt.bar(r1, train_counts, width=bar_width, label="Training Data",
            color=color1, alpha=0.85)
    plt.bar(r2, test_counts, width=bar_width, label="Testing Data",
            color=color2, alpha=0.85)
    plt.title('MNIST Digit Distribution in Datasets',
              fontsize=16, fontweight='medium', pad=20)
    plt.xlabel('Digit Class', fontsize=12, labelpad=10)
    plt.ylabel('Sample Count', fontsize=12, labelpad=10)
    plt.xticks([r + bar_width/2 for r in range(len(labels))],
               labels, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.legend(loc='upper right', frameon=True, fontsize=10, framealpha=0.8, facecolor='white')

    for i, v in enumerate(train_counts):
        plt.text(r1[i], v + (max(train_counts) * 0.01), str(v),
                ha='center', va='bottom', fontsize=9, color='#444444')
    for i, v in enumerate(test_counts):
        plt.text(r2[i], v + (max(test_counts) * 0.01), str(v),
                ha='center', va='bottom', fontsize=9, color='#444444')
    plt.tight_layout()
    plt.show()


def load_and_preprocess_data2(test_size=0.2):
    """
    Load and preprocess MNIST dataset:
    - Load data using fetch_openml
    - Normalize pixel values to [0,1]
    - Add bias term
    - Convert labels to one-hot encoding
    - Split into train/test sets
    """
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X / 255.0

    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y = y.astype(int)
    y_onehot = np.zeros((y.shape[0], 10))
    y_onehot[np.arange(y.shape[0]), y] = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=test_size, random_state=42
    )

    _, _, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test, y_train_orig, y_test_orig


def analyze_distributions(y_train, y_test):
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    train_dist = dict(zip(unique_train, counts_train))
    test_dist = dict(zip(unique_test, counts_test))

    print("Training label distribution:", train_dist)
    print("Testing label distribution:", test_dist)

    return train_dist, test_dist


def check_missing_values(X_train, X_test):
    train_missing = np.isnan(X_train).sum()
    test_missing = np.isnan(X_test).sum()

    print(f"Missing values in training set: {train_missing}")
    print(f"Missing values in testing set: {test_missing}")


# @title
def load_and_preprocess_data():
    """Load and preprocess MNIST dataset."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
  
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    return X_train, X_test, y_train, y_test

def confusion_matrix(y_true, y_pred, num_classes):
    """
    Computes the confusion matrix for a multiclass classification task.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (true) labels.
    y_pred : array-like
        Predicted labels.
    num_classes : int
        Number of classes.

    Returns:
    --------
    np.ndarray:
        Confusion matrix of shape (num_classes, num_classes),
        where cm[i, j] is the count of samples with true label i
        and predicted label j.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if np.any(y_true < 0) or np.any(y_pred < 0) or \
       np.any(y_true >= num_classes) or np.any(y_pred >= num_classes):
        raise ValueError("Labels must be in the range [0, num_classes-1].")

    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1

    return cm

def prepare_data(test_size=0.2):
    """
    Load and preprocess MNIST dataset:
    - Load data using fetch_openml
    - Normalize pixel values to [0,1]
    - Add bias term
    - Convert labels to one-hot encoding
    - Split into train/test sets
    """
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X / 255.0
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y = y.astype(int)

    y_onehot = np.zeros((y.shape[0], 10))
    y_onehot[np.arange(y.shape[0]), y] = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=test_size, random_state=42
    )

    _, _, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test, y_train_orig, y_test_orig

def plot_learning_curves(model, model_details=None):
    """
    Visualizes the learning progress of a classification model using training
    and test error curves.

    Args:
        model: The trained model object containing attributes `training_errors`
               and `test_errors` to track error rates during training.
        model_details (dict, optional): Additional details about the model, such as
                                         name and number of epochs, for annotation.
    """
    color_palette = {
        'train': '#4B0082', 
        'test': '#FF6347'   
    }
    sns.set_theme(style="whitegrid")  
    fig, ax = plt.subplots(figsize=(16, 12))

    ax.plot(model.training_errors, label='Training Error',
            linewidth=2, marker='.', markersize=4, color=color_palette['train'], alpha=0.8)
    ax.plot(model.test_errors, label='Test Error',
            linewidth=2, marker='.', markersize=4, color=color_palette['test'], alpha=0.8)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5)) 
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  

    avg_test_error = np.mean(model.test_errors)

    # Add a descriptive title and labels
    ax.set_title('MNIST Classification Learning Curves\n' +
                 f'Average Test Error: {avg_test_error:.3f}',
                 fontsize=16, fontweight='bold', color='#333333', pad=20)
    ax.set_xlabel('Iterations', fontsize=12, color='#555555')
    ax.set_ylabel('Error Rate', fontsize=12, color='#555555')

    # Add legend
    ax.legend(fontsize=10, frameon=True, facecolor='white',
             edgecolor='lightgray', loc='upper right')

    # Customize the background colors
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    if model_details:
        detail_text = (
            r"$\bf{Model\ Name:}$" + f" {model_details.get('model_name', 'PLA')}\n"
            r"$\bf{Epochs:}$" + f" {model_details.get('epochs', model.max_iter)}"
        )
        fig.text(
            0.02, 0.98, detail_text,
            ha='left', va='top', fontsize=8,
            color='#333333',
            bbox=dict(facecolor='white', edgecolor='lightgray',
                     boxstyle='round,pad=0.2', alpha=0.9),
            usetex=False
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_details=None):
    """
    Plots a confusion matrix to evaluate the classification performance visually.

    Args:
        y_true (numpy.ndarray): True labels of the dataset.
        y_pred (numpy.ndarray): Predicted labels by the model.
        model_details (dict, optional): Additional details about the model, such as
                                         name and epochs, for annotation.
    """
    sns.set_theme(style="whitegrid")  
    cm = confusion_matrix(y_true, y_pred, num_classes=10)
    accuracy = np.mean(y_true == y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))

    # A heatmap for the confusion matrix
    sns.heatmap(cm,
                cmap='RdPu',  
                square=True,  
                linewidth=0.3,  
                annot=True,  
                fmt='d',  
                annot_kws={'size': 12}) 

    plt.title('MNIST Digit Classification Results\n' +
             f'Overall Model Accuracy: {accuracy:.3f}',
             fontsize=16, fontweight='bold', color='#333333', pad=20)
    plt.xlabel('Predicted Digit Class', fontsize=12, color='#555555')
    plt.ylabel('True Digit Class', fontsize=12, color='#555555')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(width=0.5, colors='#555555')
    ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    if model_details:
        detail_text = (
            r"$\bf{Model\ Name:}$" + f" {model_details.get('model_name', 'PLA')}\n"
            r"$\bf{Epochs:}$" + f" {model_details.get('epochs', 'N/A')}"
        )
        fig.text(
            0.02, 0.98, detail_text,
            ha='left', va='top', fontsize=8,
            color='#333333',
            bbox=dict(facecolor='white', edgecolor='lightgray',
                     boxstyle='round,pad=0.2', alpha=0.9),
            usetex=False
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def print_classifier_analysis(y_true, y_pred, model_name="Model"):
    """Print detailed classification metrics for each digit"""
    # Calculate confusion matrix
    n_classes = 10
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    # Calculate metrics for each class
    precisions = np.zeros(n_classes)
    recalls = np.zeros(n_classes)
    f_scores = np.zeros(n_classes)
    accuracies = np.zeros(n_classes)

    for i in range(n_classes):
        # True Positives: diagonal elements
        tp = cm[i][i]
        # False Positives: sum of column i (predicted i) minus true positives
        fp = np.sum(cm[:, i]) - tp
        # False Negatives: sum of row i (actual i) minus true positives
        fn = np.sum(cm[i, :]) - tp

        # Calculate precision and recall
        precisions[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recalls[i] = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate F1 score (F-beta with beta=1)
        f_scores[i] = (2 * precisions[i] * recalls[i]) / (precisions[i] + recalls[i]) if (precisions[i] + recalls[i]) > 0 else 0

        total_class_samples = np.sum(cm[i, :])  # Total samples of class i
        accuracies[i] = tp / total_class_samples if total_class_samples > 0 else 0

    overall_accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    print(f"CLASSIFIER ANALYSIS FOR: \'{model_name}\':\n")

    print("PRECISION STATISTICS:")
    for i, p in enumerate(precisions):
        print(f"Digit {i} Precision: {p:.4f}")

    print("\nRECALL STATISTICS:")
    for i, r in enumerate(recalls):
        print(f"Digit {i} Recall: {r:.4f}")

    print("\nF1-SCORE METRICS BY DIGIT CLASS:")
    for i, f in enumerate(f_scores):
        print(f"Digit {i} F1-Score: {f:.4f}")

    print("\nACCURACY BY DIGIT CLASS:")
    for i, a in enumerate(accuracies):
        print(f"Digit {i} Accuracy: {a:.4f}")

    print(f"\nOVERALL MODEL PERFORMANCE:\nClassification Accuracy: {overall_accuracy:.4f}")

def plot_sensitivity_analysis(y_true, y_pred, model_name="Model"):
    """Plot sensitivity (TPR) analysis for each digit class"""
    # Calculate sensitivity for each class
    n_classes = 10
    sensitivities = np.zeros(n_classes)

    for i in range(n_classes):
        true_positives = np.sum((y_true == i) & (y_pred == i))
        false_negatives = np.sum((y_true == i) & (y_pred != i))
        sensitivities[i] = true_positives / (true_positives + false_negatives)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(n_classes), sensitivities,
                  color='skyblue', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')

    ax.set_title(f'Sensitivity (True Positive Rate) by Digit Class\n{model_name}',
                 fontsize=14, pad=20)
    ax.set_xlabel('Digit Class', fontsize=12)
    ax.set_ylabel('Sensitivity (TPR)', fontsize=12)
    ax.set_xticks(range(n_classes))
    ax.set_ylim(0, 1.1)  

    mean_sensitivity = np.mean(sensitivities)
    ax.axhline(y=mean_sensitivity, color='red', linestyle='--', alpha=0.5)
    ax.text(n_classes-1, mean_sensitivity, f'Mean: {mean_sensitivity:.3f}',
            ha='right', va='bottom', color='red')

    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return sensitivities
