from torch import nn


class ModelV0(nn.Module):
    """
    A simple feedforward neural network model consisting of fully connected layers.

    This model takes an input tensor, processes it through one hidden layer, and produces an output tensor.
    The model is initialized with the input dimension (number of features), the number of hidden layer units,
    and the output dimension (number of classes or regression targets).
    """

    def __init__(self, input_dimension: int, hidden_layer_units: int, output_dimension: int):
        """
        Initializes the feedforward neural network.

        Parameters:
        - input_dimension (int): The number of features in the input data (e.g., number of pixels in an image).
        - hidden_layer_units (int): The number of neurons in the hidden layer(s).
        - output_dimension (int): The number of output neurons (e.g., number of classes for classification).
        """
        super().__init__()

        # Define the model architecture as a sequence of layers
        self.model_architecture = nn.Sequential(

            # Flatten the input tensor into a 1D vector
            nn.Flatten(),

            # First fully connected layer (input layer to hidden layer)
            nn.Linear(in_features=input_dimension, out_features=hidden_layer_units),

            # Second fully connected layer (hidden layer to output layer)
            nn.Linear(in_features=hidden_layer_units, out_features=output_dimension)
        )

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_precision": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1_score": [],
            "test_loss": [],
            "test_acc": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1_score": [],
            "confusion_matrix": []
        }

    def forward(self, input_tensor):
        """
        Defines the forward pass of the neural network.

        This method takes an input tensor, processes it through the layers defined in the model architecture,
        and returns the output tensor.

        Parameters:
        - input_tensor (Tensor): The input data to be processed by the network.

        Returns:
        - Tensor: The output of the model after passing through the layers.
        """
        # Pass the input tensor through the defined model layers
        return self.model_architecture(input_tensor)

    def record_metric(self, metric_name: str, value: float):
        """
        Records a metric value into the history.

        Parameters:
        - metric_name (str): The name of the metric (e.g., "loss", "accuracy").
        - value (float): The value of the metric to record.
        """
        if metric_name not in self.history:
            self.history[metric_name] = []
        self.history[metric_name].append(value)

    def get_history(self, metric_name: str):
        """
        Retrieves the history of a specific metric.

        Parameters:
        - metric_name (str): The name of the metric to retrieve.

        Returns:
        - List[float]: A list of recorded values for the specified metric.
        """
        return self.history.get(metric_name, [])

    def get_all_metrics(self):
        """
        Retrieves all recorded metrics in the model's history.

        Returns:
        - Dict[str, List[float]]: A dictionary containing all recorded metrics and their values.
        """
        return self.history
