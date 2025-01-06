class MulticlassLinearRegression:
    def __init__(self):
        self.weights = None
        self.training_errors = []
        self.test_errors = []

    def fit(self, X, y):
        """
        Fit linear regression using normal equation: w = (X^T X)^(-1) X^T y
        Args:
            X: Training features (n_samples, n_features)
            y: One-hot encoded labels (n_samples, n_classes)
        """
        self.weights = np.linalg.pinv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X
        Returns class with highest regression value
        """
        scores = X @ self.weights
        return np.argmax(scores, axis=1)

    def predict_proba(self, X):
        """
        Get raw regression values (before argmax)
        Useful for calculating errors
        """
        return X @ self.weights

def evaluate_model(y_true, y_pred, model_name="Linear Regression"):
    """Calculate and print model performance metrics"""
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples

    print(f"\n{model_name} Results:")
    print(f"Classification Accuracy: {accuracy:.4f}")
    return accuracy

def plot_comparison(accuracies, model_names):
    """Plot bar chart comparing model accuracies"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=accuracies)
    plt.title("Model Comparison: Classification Accuracy", fontsize=14)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def accuracy_score(y_true, y_pred):
    """
    Calculate the classification accuracy as the proportion of correct predictions.

    Args:
        y_true: True class labels (n_samples,)
        y_pred: Predicted class labels (n_samples,)

    Returns:
        accuracy: Classification accuracy as a float
    """
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy


X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = prepare_data()

print("Training linear regression model...")
linear_regression_model = MulticlassLinearRegression()
linear_regression_model.fit(X_train, y_train)

lr_y_pred = linear_regression_model.predict(X_test)

lr_accuracy = evaluate_model(y_test_orig, lr_y_pred)

model_details = {
    'model_name': 'Linear Regression',
}
plot_confusion_matrix(y_test_orig, lr_y_pred, model_details)
print_classifier_analysis(y_test_orig, lr_y_pred, model_name="Linear Regression")
