from tqdm import tqdm
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, ConfusionMatrix
from src.utilities.utils_and_imports import check_model_numerics
import torch
from tqdm.notebook import tqdm, trange
from torchmetrics import Precision, Recall, F1Score, ConfusionMatrix


def train_model(epochs, train_loader, val_loader, model, loss_function,
                optimizer, accuracy_metric, device, num_classes, debug=False):
    """Train and validate the model, tracking metrics for both sets."""
    precision_metric = MulticlassPrecision(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    f1_score_metric = MulticlassF1Score(num_classes=num_classes, average="weighted", zero_division=0).to(device)

    if not hasattr(model, 'history'):
        model.history = {
            "train_loss": [],
            "train_acc": [],
            "train_precision": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1_score": []
        }

    for epoch in trange(epochs, desc="Overall Progress: Epochs", leave=True,
                        position=0, bar_format="{l_bar}{bar} | Batch {n_fmt}/{total_fmt}"):
        train_loss, train_acc, train_precision = 0, 0, 0
        accuracy_metric.reset()
        precision_metric.reset()

        model.train()
        for batch, (X, y) in enumerate(tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}: Training Phase",
                                            leave=False, position=1, bar_format="{l_bar}{bar} | Batch {n_fmt}/{total_fmt}")):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc += accuracy_metric(y_pred, y).item()
            train_precision += precision_metric(y_pred, y).item()

            if debug:
                print("Check model numeric (debug):")
                print("X shape:", X.shape)
                print("y shape:", y.shape)
                print("y unique values:", torch.unique(y))
                print(f"[batch={batch}]")
                check_model_numerics(model, X)
                print(f"Looked at {batch * len(X)}/{len(train_loader.dataset)} samples")
                print("-" * 50)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_precision /= len(train_loader)

        model.history["train_loss"].append(train_loss)
        model.history["train_acc"].append(train_acc)
        model.history["train_precision"].append(train_precision)

        model.eval()
        val_loss, val_acc, val_precision, val_recall, val_f1 = 0, 0, 0, 0, 0
        precision_metric.reset()
        recall_metric.reset()
        f1_score_metric.reset()
        accuracy_metric.reset()

        with torch.inference_mode():
            for X, y in tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch + 1}: Validation Phase",
                             leave=False, position=2, bar_format="{l_bar}{bar} | Batch {n_fmt}/{total_fmt}"):
                X, y = X.to(device), y.to(device)
                val_pred = model(X)
                val_loss += loss_function(val_pred, y).item()

                val_acc += accuracy_metric(val_pred, y).item()
                val_precision += precision_metric(val_pred, y).item()
                val_recall += recall_metric(val_pred, y).item()
                val_f1 += f1_score_metric(val_pred, y).item()

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            val_precision /= len(val_loader)
            val_recall /= len(val_loader)
            val_f1 /= len(val_loader)

            model.history["val_loss"].append(val_loss)
            model.history["val_acc"].append(val_acc)
            model.history["val_precision"].append(val_precision)
            model.history["val_recall"].append(val_recall)
            model.history["val_f1_score"].append(val_f1)

        print(f"\nEpoch {epoch + 1}/{epochs} Performance Report:")
        print(f"└─ [Train] Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f} | Precision: {train_precision:.2f}")
        print(f"└─ [Validation] Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f} | Precision: {val_precision:.2f} | Recall: {val_recall:.2f} | F1-Score: {val_f1:.2f}")
    print("Finished training and validation.")
    return model.history


def test_model(test_loader, model, loss_function, accuracy_metric, device, num_classes):
    """Evaluate the model on the test set."""
    precision_metric = MulticlassPrecision(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    f1_score_metric = MulticlassF1Score(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    confusion_matrix_metric = ConfusionMatrix(num_classes=num_classes, task="multiclass").to(device)

    test_loss, test_acc, test_precision, test_recall, test_f1 = 0, 0, 0, 0, 0
    y_true, y_pred = [], []

    model.eval()
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_score_metric.reset()
    confusion_matrix_metric.reset()

    with torch.inference_mode():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_function(test_pred, y).item()

            test_acc += accuracy_metric(test_pred, y).item()
            test_precision += precision_metric(test_pred, y).item()
            test_recall += recall_metric(test_pred, y).item()
            test_f1 += f1_score_metric(test_pred, y).item()
            confusion_matrix_metric.update(test_pred, y)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(test_pred.argmax(dim=1).cpu().numpy())

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        test_precision /= len(test_loader)
        test_recall /= len(test_loader)
        test_f1 /= len(test_loader)

    model.history["test_loss"].append(test_loss)
    model.history["test_acc"].append(test_acc)
    model.history["test_precision"].append(test_precision)
    model.history["test_recall"].append(test_recall)
    model.history["test_f1_score"].append(test_f1)
    model.history["confusion_matrix"].append(confusion_matrix_metric.compute().cpu().numpy())
    confusion_matrix_metric.reset()
    print(f"\n[Test] Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f} | Precision: {test_precision:.2f} | Recall: {test_recall:.2f} | F1-Score: {test_f1:.2f}")
    print("Finished test evaluation.")
