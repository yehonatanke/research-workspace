from torch import nn
import torch


class ModelV1(nn.Module):
    def __init__(self, input_dimension: int, hidden_layer_units: int, output_dimension: int):
        super().__init__()

        self.model_architecture = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dimension, hidden_layer_units),
            nn.ReLU(),
            nn.Linear(hidden_layer_units, output_dimension),
            nn.ReLU(),
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

    def forward(self, input_tensor: torch.Tensor):
        return self.model_architecture(input_tensor)

    def record_metric(self, metric_name: str, value: float):
        if metric_name not in self.history:
            self.history[metric_name] = []
        self.history[metric_name].append(value)

    def get_history(self, metric_name: str):
        return self.history.get(metric_name, [])

    def get_all_metrics(self):
        return self.history


#  -- Doesn't work well with the current implementation --
# class ModelV1(nn.Module):
#     def __init__(self, input_dimension: int, hidden_layers: list, output_dimension: int, dropout_rate: float = 0.5):
#
#         # super().__init__()
#         # self.model_architecture = nn.Sequential(
#         #     nn.Flatten(),
#         #     nn.Linear(in_features=input_dimension, out_features=hidden_layer_units),
#         #     nn.Linear(in_features=hidden_layer_units, out_features=output_dimension)
#         # )
#         super().__init__()
#         layers = [nn.Flatten()]
#
#         prev_units = input_dimension
#         for units in hidden_layers:
#             layers.extend([
#                 nn.Linear(in_features=prev_units, out_features=units),
#                 nn.ReLU(),
#                 nn.Dropout(p=dropout_rate)
#             ])
#             prev_units = units
#
#         # Final output layer
#         layers.append(nn.Linear(in_features=prev_units, out_features=output_dimension))
#
#         self.model_architecture = nn.Sequential(*layers)
#
#         self.history = {
#             "train_loss": [],
#             "train_acc": [],
#             "train_precision": [],
#             "val_loss": [],
#             "val_acc": [],
#             "val_precision": [],
#             "val_recall": [],
#             "val_f1_score": [],
#             "test_loss": [],
#             "test_acc": [],
#             "test_precision": [],
#             "test_recall": [],
#             "test_f1_score": [],
#             "confusion_matrix": []
#         }
#
#     def forward(self, input_tensor):
#         return self.model_architecture(input_tensor)
#
#     def record_metric(self, metric_name: str, value: float):
#         if metric_name not in self.history:
#             self.history[metric_name] = []
#         self.history[metric_name].append(value)
#
#     def get_history(self, metric_name: str):
#         return self.history.get(metric_name, [])
#
#     def get_all_metrics(self):
#         return self.history
