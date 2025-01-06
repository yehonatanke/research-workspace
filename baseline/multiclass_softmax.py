"""
Algorithm Overview:
-----------------
Softmax regression extends logistic regression to handle multiple classes by using the softmax function
to convert raw model outputs into probability distributions over the classes.

Key Components:
1. Softmax Function: σ(z)_i = exp(z_i) / Σ_j exp(z_j)
   - Converts raw scores into probabilities that sum to 1
   - Numerically stabilized by subtracting max value before exponential

2. Cross-Entropy Loss: -Σ y_i log(p_i)
   - Measures difference between predicted probabilities and true labels
   - Using one-hot encoded ground truth labels

3. Gradient Descent:
   - Computes gradient of loss with respect to weights
   - Updates weights iteratively: w = w - η∇L
   - Uses mini-batches for better computational efficiency
"""
class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, num_epochs=50, batch_size=128):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weights = None

    def load_mnist(self):
        """
        Loads and preprocesses the MNIST dataset.

        Returns:
        --------
        tuple: (X_train, X_test, y_train, y_test)
            - X_train, X_test: Features with added bias term and normalized pixels
            - y_train, y_test: Integer labels for digits
        """
        mnist = fetch_openml('mnist_784', version=1)
        X, y = mnist['data'].to_numpy(), mnist['target'].to_numpy().astype(int)
        X = X / 255.0
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def one_hot_encode(y, num_classes=10):
        return np.eye(num_classes)[y]

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_loss(self, X, y_encoded):
        z = np.dot(X, self.weights.T)
        probs = self.softmax(z)
        return -np.mean(np.sum(y_encoded * np.log(probs + 1e-10), axis=1))

    def compute_gradient(self, X, y_encoded):
        z = np.dot(X, self.weights.T)
        probs = self.softmax(z)
        grad = np.dot(X.T, (probs - y_encoded)).T / X.shape[0]
        return grad

    def predict(self, X):
        z = np.dot(X, self.weights.T)
        probs = self.softmax(z)
        return np.argmax(probs, axis=1)

    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    def train(self, X_train, y_train, X_test, y_test):
        num_features = X_train.shape[1]
        num_classes = 10
        self.weights = np.random.randn(num_classes, num_features) * 0.01

        y_train_encoded = self.one_hot_encode(y_train)
        y_test_encoded = self.one_hot_encode(y_test)

        num_batches = len(X_train) // self.batch_size
        train_losses = []
        test_losses = []

        progress_bar = tqdm(
            range(self.num_epochs),
            desc="Training Progress",
            unit="epoch",
            leave=False,
            position=0,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        )

        for epoch in progress_bar:
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train_encoded[indices]

            epoch_loss = 0
            for batch in range(num_batches):
                start_idx = batch * self.batch_size
                end_idx = start_idx + self.batch_size

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                grad = self.compute_gradient(X_batch, y_batch)
                self.weights -= self.learning_rate * grad

                batch_loss = self.compute_loss(X_batch, y_batch)
                epoch_loss += batch_loss

            avg_epoch_loss = epoch_loss / num_batches
            train_losses.append(avg_epoch_loss)

            test_loss = self.compute_loss(X_test, y_test_encoded)
            test_losses.append(test_loss)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}, Train Loss: {avg_epoch_loss:.4f}, Test Loss: {test_loss:.4f}")

        return train_losses, test_losses

    def calculate_confusion_and_sensitivity(self, y_true, y_pred):
        """
        Computes the confusion matrix for each digit and sensitivity (TPR) for each class.

        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels

        Returns:
        --------
        tuple: (confusion_matrix, sensitivities)
            confusion_matrix : numpy.ndarray
                Confusion matrix for the classification results.
            sensitivities : list
                Sensitivity values for each digit (TPR).
        """
        cm = confusion_matrix(y_true, y_pred, num_classes=10)
        sensitivities = []
        for i in range(10):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            sensitivities.append(sensitivity)
        return cm, sensitivities

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, num_classes=10, num_epochs=50, lr=0):
        """
        Plots a confusion matrix to evaluate classification performance.

        Args:
            y_true (numpy.ndarray): True labels of the dataset.
            y_pred (numpy.ndarray): Predicted labels by the model.
            model_details (dict, optional): Details about the model for annotation.
            num_classes (int, optional): Number of classes for confusion matrix.
        """
        model_details = {
            'model_name': 'Softmax Regression',
            'learning_rate': lr,
            'epochs': num_epochs
            }

        sns.set_theme(style="whitegrid")

        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred, num_classes=10)

        # Calculate model accuracy
        accuracy = np.mean(y_true == y_pred)

        # Initialize figure and axis
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create a heatmap for the confusion matrix
        sns.heatmap(cm, cmap='RdPu', square=True, linewidth=0.3, annot=True, fmt='d',
                    annot_kws={'size': 12})

        # Add titles and labels
        plt.title('Classification Results\n' +
                  f'Accuracy: {accuracy:.2%}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)

        # Display the model details
        if model_details:
            detail_text = (
                r"$\bf{Model:}$" + f" {model_details.get('model_name', 'N/A')}\n" +
                r"$\bf{Learning\ Rate:}$" + f" {model_details.get('learning_rate', 'N/A')}\n" +
                r"$\bf{Epochs:}$" + f" {model_details.get('epochs', 'N/A')}"
            )
            fig.text(
                0.02, 0.98, detail_text,
                ha='left', va='top', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='lightgray',
                          boxstyle='round,pad=0.3', alpha=0.9)
            )

        # Enhance visual appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Show the plot
        plt.show()

    @staticmethod
    def calculate_sensitivity(y_true, y_pred):
        """
        Calculates sensitivity (TPR) for each class.

        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels

        Returns:
        --------
        list: Sensitivity values for each digit
        """
        cm = confusion_matrix(y_true, y_pred, num_classes=10)
        sensitivities = []
        for i in range(10):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            sensitivity = tp / (tp + fn)
            sensitivities.append(sensitivity)
        return sensitivities


    def plot_loss(self, train_losses, test_losses, model_details=None, num_epochs=50, lr=0):
        """
        Visualizes the training and test loss over epochs with a customized style.

        Args:
            train_losses: List of training loss values.
            test_losses: List of test loss values.
            model_details (dict, optional): Additional details about the model, such as
                                            name and number of epochs, for annotation.
        """
        model_details = {
            'model_name': 'Softmax Regression',
            'learning_rate': lr,
            'epochs': num_epochs
            }
        # Set a custom color palette and grid-based visual theme
        color_palette = {
            'train': '#4B0082',  # Indigo for training loss
            'test': '#FF6347'    # Tomato for test loss
        }
        sns.set_theme(style="whitegrid")  # Apply a white grid theme for improved readability

        # Initialize the figure and axis with dimensions
        fig, ax = plt.subplots(figsize=(16, 12))

        # Generate the epochs range
        epochs = range(1, len(train_losses) + 1)

        # Plot training and test loss curves
        ax.plot(epochs, train_losses, label='Training Loss',
                linewidth=2, marker='.', markersize=4, color=color_palette['train'], alpha=0.8)
        ax.plot(epochs, test_losses, label='Test Loss',
                linewidth=2, marker='.', markersize=4, color=color_palette['test'], alpha=0.8)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Limit number of x-axis ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Limit number of y-axis ticks

        # Calculate average test loss to include in the title
        avg_test_loss = np.mean(test_losses)

        # Add a descriptive title and labels
        ax.set_title('Training and Test Loss over Epochs\n' +
                    f'Average Test Loss: {avg_test_loss:.3f}',
                    fontsize=16, fontweight='bold', color='#333333', pad=20)
        ax.set_xlabel('Epochs', fontsize=12, color='#555555')
        ax.set_ylabel('Loss', fontsize=12, color='#555555')

        # Add legend
        ax.legend(frameon=True, facecolor='white',
                  edgecolor='lightgray', loc='upper right')

        # Customize the background colors
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Display the model details
        if model_details:
            detail_text = (
                r"$\bf{Model\ Name:}$" + f" {model_details.get('model_name', 'N/A')}\n"
                r"$\bf{Learning\ Rate:}$" + f" {model_details.get('learning_rate', 'N/A')}\n" +
                r"$\bf{Epochs:}$" + f" {model_details.get('epochs', len(epochs))}"
            )
            fig.text(
                0.02, 0.98, detail_text,
                ha='left', va='top', fontsize=10,
                color='#333333',
                bbox=dict(facecolor='white', edgecolor='lightgray',
                          boxstyle='round,pad=0.4', alpha=0.9),
                usetex=False
            )

        # Adjust layout to prevent overlap and display the plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def run_model(self):
        X_train, X_test, y_train, y_test = self.load_mnist()
        train_losses, test_losses = self.train(X_train, y_train, X_test, y_test)

        y_pred = self.predict(X_test)
        test_accuracy = self.calculate_accuracy(y_test, y_pred)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        conf_matrix, sensitivities = model.calculate_confusion_and_sensitivity(y_test, y_pred)

        self.plot_loss(train_losses, test_losses, num_epochs=self.num_epochs, lr=self.learning_rate)
        self.plot_confusion_matrix(y_test, y_pred, num_epochs=self.num_epochs)

        # sensitivities = self.calculate_sensitivity(y_test, y_pred)
        # for digit, sens in enumerate(sensitivities):
        #     print(f"Sensitivity for digit {digit}: {sens:.4f}")

        cm, sensitivities = self.calculate_confusion_and_sensitivity(y_test, y_pred)
        print("\nConfusion Matrix:")
        # print(cm) # print confusion matrix without a plot
        print("\nSensitivities (TPR) for each digit:")
        for digit, sens in enumerate(sensitivities):
            print(f"Digit {digit}: {sens:.4f}")

# Run model to observe initial performance; further analysis will follow later
model = SoftmaxRegression(learning_rate=0.001, num_epochs=100)
model.run_model()


# Define a list of different hyperparameter configurations to test
# Each configuration is a dictionary containing the learning rate, number of epochs, and batch size
configs = [
    {"learning_rate": 0.1, "num_epochs": 100, "batch_size": 128},  
    {"learning_rate": 0.05, "num_epochs": 100, "batch_size": 64}, 
    {"learning_rate": 0.01, "num_epochs": 100, "batch_size": 128}, 
    {"learning_rate": 0.2, "num_epochs": 50, "batch_size": 256}  
]

results = {}

progress_bar = tqdm(
    total=len(configs),  # Total number of iterations
    desc="Processing Configurations",  # Description of the task
    unit="config",  # Unit of measurement
    leave=True,  # Leave the progress bar displayed after completion
    position=0,  # Position of the progress bar (useful in multi-threaded applications)
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
)

# Iterate through each configuration and run the model with those hyperparameters
for idx, config in enumerate(configs):
    progress_bar.set_description(f"Processing Model {idx + 1} with config: {config}")  # Update description dynamically

    print(f"\nRunning Model {idx + 1} with config: {config}")  # Log the configuration being tested
    model = SoftmaxRegression(
        learning_rate=config["learning_rate"],  # Set the learning rate
        num_epochs=config["num_epochs"],  # Set the number of epochs
        batch_size=config["batch_size"]  # Set the batch size
    )

    # Load the MNIST dataset
    X_train, X_test, y_train, y_test = model.load_mnist()

    # Train the model and get the training and test losses
    train_losses, test_losses = model.train(X_train, y_train, X_test, y_test)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = model.calculate_accuracy(y_test, y_pred)

    # Calculate the confusion matrix and sensitivities
    conf_matrix, sensitivities = model.calculate_confusion_and_sensitivity(y_test, y_pred)

    # Store the results of the current configuration
    results[idx] = {
        "config": config,
        "train_losses": train_losses, 
        "test_losses": test_losses,  
        "accuracy": accuracy,  
        "confusion_matrix": conf_matrix,  
        "sensitivities": sensitivities  
    }

    # Log the accuracy and sensitivities of the current model
    print(f"Accuracy for Model {idx + 1}: {accuracy:.4f}")
    print(f"Sensitivities for Model {idx + 1}: {['{:.4f}'.format(s) for s in sensitivities]}")

    # Plot the loss curves and confusion matrix for the current model
    model.plot_loss(train_losses, test_losses, num_epochs=config["num_epochs"], lr=config["learning_rate"])
    model.plot_confusion_matrix(y_test, y_pred, num_epochs=config["num_epochs"], lr=config["learning_rate"])
    progress_bar.update(1)

# Compare the results of all models
print("\nComparison of Models:")
for idx, result in results.items():
    config = result["config"]  # Get the configuration of the model
    accuracy = result["accuracy"]  # Get the accuracy of the model
    avg_train_loss = np.mean(result["train_losses"])  # Calculate the average training loss
    avg_test_loss = np.mean(result["test_losses"])  # Calculate the average test loss
    sensitivities = result["sensitivities"]  # Get the sensitivities of the model

    # Log the comparison results
    print(f"Model {idx + 1} | Config: {config} | Accuracy: {accuracy:.4f} | "
          f"Avg Train Loss: {avg_train_loss:.4f} | Avg Test Loss: {avg_test_loss:.4f}")
    print(f"   Sensitivities: {[f'{s:.4f}' for s in sensitivities]}")
