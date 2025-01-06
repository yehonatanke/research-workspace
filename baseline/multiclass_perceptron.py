class MulticlassPerceptron_V1:
    """
    A multiclass perceptron implementation using the Pocket Algorithm.

    Attributes:
        n_classes (int): Number of target classes. Default is 10.
        max_iter (int): Maximum number of iterations for training. Default is 1000.
        weights (numpy.ndarray): Current weights for each class.
        pocket_weights (numpy.ndarray): Best weights (least error) retained for each class.
        best_errors (numpy.ndarray): Minimum errors recorded for each class.
        training_errors (list): List to track training error per epoch.
        test_errors (list): List to track test error per epoch.
        pocket_weights_history (list): List to track pocket weights per epoch.
    """
    def __init__(self, n_classes=10, max_iter=1000):
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.weights = None  # To store current weights for each class
        self.pocket_weights = None  # To store best weights based on error
        self.best_errors = None  # To track the lowest errors for each class
        self.training_errors = []  # To log training error for each epoch
        self.test_errors = []  # To log test error for each epoch
        self.pocket_weights_history = []  # To log pocket weights for each epoch

    def _init_weights(self, n_features):
        """
        Initializes the weight matrices and pocket weights.

        Args:
            n_features (int): Number of features in the input data.
        """
        # Initialize weights to zero for all classes and features
        # Creates a matrix of shape (number of classes, number of features)
        self.weights = np.zeros((self.n_classes, n_features))

        # Pocket weights start as a copy of the initial weights
        self.pocket_weights = self.weights.copy()

        # Initial best errors set to 1.0 (worst case scenario - all misclassified)
        # This is an array with one element per class
        self.best_errors = np.ones(self.n_classes)

    def _get_binary_labels(self, y):
        """
        Converts one-hot encoded labels into binary labels for all classes simultaneously.

        Args:
            y (numpy.ndarray): One-hot encoded labels.

        Returns:
            numpy.ndarray: Binary labels of shape (n_samples, n_classes).
        """
        return 2 * y - 1  # Convert 1 to +1 and 0 to -1 for all classes

    def _calculate_errors(self, X, y, weights):
        """
        Calculates the classification errors for all classes simultaneously.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): One-hot encoded labels.
            weights (numpy.ndarray): Weights used for prediction.

        Returns:
            numpy.ndarray: Fraction of misclassified samples per class.
        """
        binary_y = self._get_binary_labels(y) # Binary labels for current class
        scores = X @ weights.T # Compute scores as dot product of X and weights
        predictions = np.where(scores > 0, 1, -1) # Predict class based on sign of scores
        return np.mean(predictions != binary_y, axis=0) # Calculate proportion of predictions

    def fit(self, X_train, y_train, X_test, y_test):
        """
        Trains the perceptron model using the Pocket Algorithm.

        Args:
            X_train (numpy.ndarray): Training data matrix of shape (n_samples, n_features).
            y_train (numpy.ndarray): One-hot encoded training labels.
            X_test (numpy.ndarray): Test data matrix.
            y_test (numpy.ndarray): One-hot encoded test labels.
        """
        n_samples, n_features = X_train.shape # Get number of samples and features
        self._init_weights(n_features) # Initialize weights

        progress_bar = tqdm(range(self.max_iter), desc="Training Progress", unit="epoch", leave=True, position=0)

        for iteration in progress_bar:
            # Compute scores and determine misclassified samples for all classes
            binary_y_train = self._get_binary_labels(y_train) # Binary labels for the current class
            scores = X_train @ self.weights.T
            misclassified = (scores * binary_y_train <= 0)  # Shape: (n_samples, n_classes)

            # Nested progress bar for processing each class
            class_progress = tqdm(range(self.n_classes), desc="Processing Classes", unit="class", leave=False, position=1)

            # Update weights for misclassified samples
            for class_idx in class_progress:
                if np.any(misclassified[:, class_idx]):
                    update_idx = np.where(misclassified[:, class_idx])[0][0]
                    self.weights[class_idx] += binary_y_train[update_idx, class_idx] * X_train[update_idx]

            # Calculate errors for all classes
            current_errors = self._calculate_errors(X_train, y_train, self.weights)
            improved_mask = current_errors < self.best_errors

            # Update pocket weights where errors improved
            self.pocket_weights[improved_mask] = self.weights[improved_mask].copy()
            self.best_errors[improved_mask] = current_errors[improved_mask]

            # Calculate and log training and test errors
            avg_train_error = np.mean(self.best_errors)
            avg_test_error = np.mean(self._calculate_errors(X_test, y_test, self.pocket_weights))

            self.training_errors.append(avg_train_error)
            self.test_errors.append(avg_test_error)

            if iteration % 100 == 0:
                self.pocket_weights_history.append(self.pocket_weights.copy())

            progress_bar.set_postfix({'Train Error': f'{avg_train_error:.4f}', 'Test Error': f'{avg_test_error:.4f}'})

    def predict(self, X):
        """
        Predicts the class labels for input samples using the best weights.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted class labels.
        """
        scores = X @ self.pocket_weights.T # Compute scores for all classes
        return np.argmax(scores, axis=1) # Select class with highest score

    def calculate_sensitivity(self, X, y_true):
        """
        Calculates sensitivity (TPR) for each class.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).
            y_true (numpy.ndarray): True class labels as a 1D array.

        Returns:
            numpy.ndarray: Sensitivity values for each class.
        """
        y_pred = self.predict(X)
        sensitivities = np.zeros(self.n_classes)

        for i in range(self.n_classes):
            true_positives = np.sum((y_true == i) & (y_pred == i))
            false_negatives = np.sum((y_true == i) & (y_pred != i))
            sensitivities[i] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        return sensitivities


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

    # Normalize features to [0,1]
    X = X / 255.0

    # Add bias term
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Convert labels to integers
    y = y.astype(int)

    # Convert to one-hot encoding
    y_onehot = np.zeros((y.shape[0], 10))
    y_onehot[np.arange(y.shape[0]), y] = 1

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=test_size, random_state=42
    )

    # Also return original labels for confusion matrix
    _, _, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test, y_train_orig, y_test_orig


# Prepare the data again in an orderly manner to ensure correct loading of the data
X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = prepare_data()

# Set max_iter=500 as testing showed no significant improvement beyond this point.
max_iter = 500
pla_model_name = "Perceptron Learning Algorithm V1"

# Model details (for the plots)
pla_model_details = {
    'model_name': pla_model_name,
    'epochs': max_iter
}

# Create and train model
pla_model = MulticlassPerceptron_V1(max_iter=max_iter)

print("Training model...")
pla_model.fit(X_train, y_train, X_test, y_test)

# Make predictions
pla_y_pred = pla_model.predict(X_test)
