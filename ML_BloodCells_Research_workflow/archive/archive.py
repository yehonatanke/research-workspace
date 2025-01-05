from sklearn.metrics import classification_report
from tqdm import tqdm
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, ConfusionMatrix
import torch

from src.utilities.utils_and_imports import check_model_numerics
import torch
from tqdm.notebook import tqdm, trange
from torchmetrics import Precision, Recall, F1Score, ConfusionMatrix


def train_and_evaluate(epochs, train_loader, val_loader, test_loader, model, loss_function,
                       optimizer, accuracy_metric, device, num_classes, class_names, debug=False):
    # Initialize history with confusion_matrix key
    model.history = {
        "train_loss": [], "train_acc": [], "train_precision": [],
        "val_loss": [], "val_acc": [], "val_precision": [],
        "val_recall": [], "val_f1_score": [],
        "test_loss": [], "test_acc": [], "test_precision": [],
        "test_recall": [], "test_f1_score": [],
        "confusion_matrix": []
    }

    # Create metrics with proper reset functionality
    precision_metric = MulticlassPrecision(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    f1_score_metric = MulticlassF1Score(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    confusion_matrix_metric = ConfusionMatrix(num_classes=num_classes, task="multiclass").to(device)

    for epoch in tqdm(range(epochs)):
        # Reset metrics at the start of each epoch
        train_loss, train_acc, train_precision = 0, 0, 0

        # Training Loop
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            y_pred = model(X)
            loss = loss_function(y_pred, y)

            loss.backward()
            optimizer.step()

            # Detach predictions for metric calculation to prevent gradient tracking
            with torch.no_grad():
                train_loss += loss.item()

                # Evaluate metrics on the entire batch
                batch_acc = accuracy_metric(y_pred.detach(), y).item()
                batch_precision = precision_metric(y_pred.detach(), y).item()

                train_acc += batch_acc
                train_precision += batch_precision

        # Average training metrics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_precision /= len(train_loader)

        # Record training metrics
        model.history["train_loss"].append(train_loss)
        model.history["train_acc"].append(train_acc)
        model.history["train_precision"].append(train_precision)

        # Validation Loop with unified evaluation function
        def evaluate_model(data_loader, confusion_matrix_metric=None, is_test=False):

            # Reset metrics before evaluation
            precision_metric.reset()
            recall_metric.reset()
            f1_score_metric.reset()

            loss, acc, precision, recall, f1 = 0, 0, 0, 0, 0

            model.eval()
            with torch.inference_mode():
                for X, y in data_loader:
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)

                    loss += loss_function(y_pred, y).item()

                    # Compute metrics
                    acc += accuracy_metric(y_pred, y).item()
                    precision += precision_metric(y_pred, y).item()
                    recall += recall_metric(y_pred, y).item()
                    f1 += f1_score_metric(y_pred, y).item()

                    if is_test:
                        # Update confusion matrix
                        confusion_matrix_metric.update(y_pred, y)

                # Average metrics to ensures that the metrics reflect the performance over the entire dataset
                loss /= len(data_loader)
                acc /= len(data_loader)
                precision /= len(data_loader)
                recall /= len(data_loader)
                f1 /= len(data_loader)

                # Get confusion matrix
                # conf_matrix = confusion_matrix_metric.compute().cpu().numpy()
                if is_test:
                    return loss, acc, precision, recall, f1, confusion_matrix_metric
                else:
                    return loss, acc, precision, recall, f1

        # Validation evaluation
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(val_loader, is_test=False)

        # Record validation metrics
        model.history["val_loss"].append(val_loss)
        model.history["val_acc"].append(val_acc)
        model.history["val_precision"].append(val_precision)
        model.history["val_recall"].append(val_recall)
        model.history["val_f1_score"].append(val_f1)

        # Test evaluation
        test_loss, test_acc, test_precision, test_recall, test_f1, confusion_matrix_metric = evaluate_model(test_loader, confusion_matrix_metric, is_test=True)

        # Record test metrics
        model.history["test_loss"].append(test_loss)
        model.history["test_acc"].append(test_acc)
        model.history["test_precision"].append(test_precision)
        model.history["test_recall"].append(test_recall)
        model.history["test_f1_score"].append(test_f1)

        # Print epoch summary
        print(f"\nEpoch: {epoch}")
        print(f"[Loss] Train: {train_loss:.4f} | Validation: {val_loss:.4f} | Test: {test_loss:.4f}")
        print(f"[Accuracy] Train: {train_acc * 100:.2f}% | Validation: {val_acc * 100:.2f}% | Test: {test_acc * 100:.2f}%")
        print(f"[Precision] Train: {train_precision * 100:.2f}% | Validation: {val_precision * 100:.2f}% | Test: {test_precision * 100:.2f}%")
        print(f"[Recall] Validation: {val_recall * 100:.2f}% | Test: {test_recall * 100:.2f}%")
        print(f"[F1-Score] Validation: {val_f1 * 100:.2f}% | Test: {test_f1 * 100:.2f}%")

    conf_matrix = confusion_matrix_metric.compute().cpu().numpy()
    model.history["confusion_matrix"].append(conf_matrix)
    confusion_matrix_metric.reset()

    return model.history


def train_and_evaluateGPT(epochs, train_loader, val_loader, test_loader, model, loss_function,
                          optimizer, accuracy_metric, device, num_classes, class_names, debug=False):
    precision_metric = MulticlassPrecision(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    f1_score_metric = MulticlassF1Score(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    confusion_matrix_metric = ConfusionMatrix(num_classes=num_classes, task="multiclass").to(device)

    if not hasattr(model, 'history'):
        model.history = {
            "train_loss": [], "train_acc": [], "train_precision": [],
            "val_loss": [], "val_acc": [], "val_precision": [],
            "val_recall": [], "val_f1_score": [],
            "test_loss": [], "test_acc": [], "test_precision": [],
            "test_recall": [], "test_f1_score": [],
            "confusion_matrix": []
        }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_precision = 0, 0, 0

        # Training Loop
        model.train()
        accuracy_metric.reset()  # Reset accuracy metric at the start of each epoch
        precision_metric.reset()  # Reset precision metric at the start of each epoch
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc += accuracy_metric(y_pred, y).item()
            train_precision += precision_metric(y_pred, y).item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_precision /= len(train_loader)

        model.history["train_loss"].append(train_loss)
        model.history["train_acc"].append(train_acc)
        model.history["train_precision"].append(train_precision)

        # Validation Loop
        model.eval()
        # accuracy_metric.reset()  # Reset metrics for validation
        # precision_metric.reset()
        # recall_metric.reset()
        # f1_score_metric.reset()

        val_loss, val_acc, val_precision, val_recall, val_f1 = 0, 0, 0, 0, 0
        with torch.inference_mode():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_pred = model(X)
                val_loss += loss_function(val_pred, y).item()

                # Compute metrics for validation
                val_acc += accuracy_metric(val_pred, y).item()
                val_precision += precision_metric(val_pred, y).item()
                val_recall += recall_metric(val_pred, y).item()
                val_f1 += f1_score_metric(val_pred, y).item()
                # confusion_matrix_metric.update(val_pred, y)

            # Average validation metrics over the entire loader
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

        # Testing Loop
        # accuracy_metric.reset()
        # precision_metric.reset()
        # recall_metric.reset()
        # f1_score_metric.reset()
        # confusion_matrix_metric.reset()
        y_true = []
        y_pred = []
        test_loss, test_acc, test_precision, test_recall, test_f1 = 0, 0, 0, 0, 0
        with torch.inference_mode():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                test_pred = model(X)
                test_loss += loss_function(test_pred, y).item()

                # Compute metrics for testing
                test_acc += accuracy_metric(test_pred, y).item()
                test_precision += precision_metric(test_pred, y).item()
                test_recall += recall_metric(test_pred, y).item()
                test_f1 += f1_score_metric(test_pred, y).item()
                confusion_matrix_metric.update(test_pred, y)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(test_pred.argmax(dim=1).cpu().numpy())

            # Average testing metrics over the entire loader
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
        # confusion_matrix_metric.reset()

        print(f"\nEpoch: {epoch}")
        print(f"[Loss] Train: {train_loss:.4f} | Validation: {val_loss:.4f} | Test: {test_loss:.4f}")
        print(f"[Accuracy] Train: {train_acc * 100:.2f}% | Validation: {val_acc * 100:.2f}% | Test: {test_acc * 100:.2f}%")
        print(f"[Precision] Train: {train_precision * 100:.2f}% | Validation: {val_precision * 100:.2f}% | Test: {test_precision * 100:.2f}%")
        print(f"[Recall] Validation: {val_recall * 100:.2f}% | Test: {test_recall * 100:.2f}%")
        print(f"[F1-Score] Validation: {val_f1 * 100:.2f}% | Test: {test_f1 * 100:.2f}%")

    return model.history


def train_and_evaluate_firstVERSION(epochs, train_loader, val_loader, test_loader, model, loss_function,
                                    optimizer, accuracy_metric, device, num_classes, class_names, debug=False):
    # Initialize metrics
    precision_metric = MulticlassPrecision(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    f1_score_metric = MulticlassF1Score(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    confusion_matrix_metric = ConfusionMatrix(num_classes=num_classes, task="multiclass").to(device)

    # Ensure the model has a history attribute
    if not hasattr(model, 'history'):
        model.history = {
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

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_precision = 0, 0, 0

        # Training Loop
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy_metric(torch.argmax(y_pred, dim=1), y).item()
            precision_metric(y_pred, y).item()

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
        train_acc = precision_metric.compute()
        train_precision /= len(train_loader)

        model.history["train_loss"].append(train_loss)
        model.history["train_acc"].append(train_acc)
        model.history["train_precision"].append(train_precision)
        precision_metric.reset()

        # Validation Loop
        model.eval()
        val_loss, val_acc, val_precision, val_recall, val_f1 = 0, 0, 0, 0, 0
        with torch.inference_mode():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_pred = model(X)
                val_loss += loss_function(val_pred, y).item()

                val_acc += accuracy_metric(torch.argmax(val_pred, dim=1), y).item()
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

        # Testing Loop
        y_true = []
        y_pred = []
        test_loss, test_acc, test_precision, test_recall, test_f1 = 0, 0, 0, 0, 0
        with torch.inference_mode():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                test_pred = model(X)
                test_loss += loss_function(test_pred, y).item()

                # Compute metrics
                test_acc += accuracy_metric(torch.argmax(test_pred, dim=1), y).item()
                test_precision += precision_metric(test_pred, y).item()
                test_recall += recall_metric(test_pred, y).item()
                test_f1 += f1_score_metric(test_pred, y).item()
                confusion_matrix_metric.update(test_pred, y)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(test_pred.argmax(dim=1).cpu().numpy())

            # Average testing metrics over the entire loader
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
        # confusion_matrix_metric.reset()

        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

        print(f"\nEpoch: {epoch}")
        print(f"[Loss] Train: {train_loss:.4f} | Validation: {val_loss:.4f} | Test: {test_loss:.4f}")
        print(f"[Accuracy] Train: {train_acc * 100:.2f}% | Validation: {val_acc * 100:.2f}% | Test: {test_acc * 100:.2f}%")
        print(f"[Precision] Train: {train_precision * 100:.2f}% | Validation: {val_precision * 100:.2f}% | Test: {test_precision * 100:.2f}%")
        print(f"[Recall] Validation: {val_recall * 100:.2f}% | Test: {test_recall * 100:.2f}%")
        print(f"[F1-Score] Validation: {val_f1 * 100:.2f}% | Test: {test_f1 * 100:.2f}%")

    print("Training and evaluation completed successfully ------------------\n\n")
    return model.history


def train_and_evaluateClaude(epochs, train_loader, val_loader, test_loader, model, loss_function,
                             optimizer, accuracy_metric, device, num_classes, class_names, debug=False):
    # Initialize metrics
    precision_metric = MulticlassPrecision(
        num_classes=num_classes,
        average="weighted",
        zero_division=0
    ).to(device)
    recall_metric = MulticlassRecall(
        num_classes=num_classes,
        average="weighted",
        zero_division=0
    ).to(device)
    f1_score_metric = MulticlassF1Score(
        num_classes=num_classes,
        average="weighted",
        zero_division=0
    ).to(device)
    confusion_matrix_metric = ConfusionMatrix(
        num_classes=num_classes,
        task="multiclass"
    ).to(device)

    # Ensure the model has a history attribute
    if not hasattr(model, 'history'):
        model.history = {
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

    for epoch in tqdm(range(epochs)):
        # Reset metrics at the start of each epoch
        train_loss, train_acc = 0.0, 0.0
        precision_metric.reset()
        recall_metric.reset()
        f1_score_metric.reset()

        # Training Loop
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(X)

            # Compute loss
            loss = loss_function(y_pred, y)
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Debug if needed
            if debug:
                print("Check model numeric (debug):")
                print("X shape:", X.shape)
                print("y shape:", y.shape)
                print("y unique values:", torch.unique(y))
                print(f"[batch={batch}]")
                check_model_numerics(model, X)
                print(f"Looked at {batch * len(X)}/{len(train_loader.dataset)} samples")
                print("-" * 50)

        # Compute training metrics
        train_loss /= len(train_loader)

        # Validation Loop
        model.eval()
        val_metrics = {
            'loss': 0.0,
            'acc': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

        y_true_val, y_pred_val = [], []

        with torch.inference_mode():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_pred = model(X)

                # Compute metrics
                val_metrics['loss'] += loss_function(val_pred, y).item()

                # Get predictions
                preds = torch.argmax(val_pred, dim=1)

                # Store for later use
                y_true_val.extend(y.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())

                # Update metrics
                accuracy_metric(preds, y)
                precision_metric(val_pred, y)
                recall_metric(val_pred, y)
                f1_score_metric(val_pred, y)

            # Compute average metrics
            for key in ['loss']:
                val_metrics[key] /= len(val_loader)

        # Compute final validation metrics
        val_metrics['acc'] = accuracy_metric.compute().item()
        val_metrics['precision'] = precision_metric.compute().item()
        val_metrics['recall'] = recall_metric.compute().item()
        val_metrics['f1'] = f1_score_metric.compute().item()

        # Testing Loop
        test_metrics = {
            'loss': 0.0,
            'acc': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

        y_true_test, y_pred_test = [], []

        with torch.inference_mode():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                test_pred = model(X)

                # Compute metrics
                test_metrics['loss'] += loss_function(test_pred, y).item()

                # Get predictions
                preds = torch.argmax(test_pred, dim=1)

                # Store for later use
                y_true_test.extend(y.cpu().numpy())
                y_pred_test.extend(preds.cpu().numpy())

                # Update confusion matrix
                confusion_matrix_metric.update(test_pred, y)

            # Compute average metrics
            test_metrics['loss'] /= len(test_loader)

            # Compute final test metrics
            test_metrics['acc'] = accuracy_metric(
                torch.tensor(y_pred_test).to(device),
                torch.tensor(y_true_test).to(device)
            ).item()
            test_metrics['precision'] = precision_metric(
                torch.tensor(y_pred_test).to(device),
                torch.tensor(y_true_test).to(device)
            ).item()
            test_metrics['recall'] = recall_metric(
                torch.tensor(y_pred_test).to(device),
                torch.tensor(y_true_test).to(device)
            ).item()
            test_metrics['f1'] = f1_score_metric(
                torch.tensor(y_pred_test).to(device),
                torch.tensor(y_true_test).to(device)
            ).item()

        # Store metrics in model history
        model.history['train_loss'].append(train_loss)
        model.history['val_loss'].append(val_metrics['loss'])
        model.history['val_acc'].append(val_metrics['acc'])
        model.history['val_precision'].append(val_metrics['precision'])
        model.history['val_recall'].append(val_metrics['recall'])
        model.history['val_f1_score'].append(val_metrics['f1'])
        model.history['test_loss'].append(test_metrics['loss'])
        model.history['test_acc'].append(test_metrics['acc'])
        model.history['test_precision'].append(test_metrics['precision'])
        model.history['test_recall'].append(test_metrics['recall'])
        model.history['test_f1_score'].append(test_metrics['f1'])

        # Compute and store confusion matrix
        confusion_matrix = confusion_matrix_metric.compute()
        model.history['confusion_matrix'].append(confusion_matrix.cpu().numpy())

        # Print detailed metrics
        print(f"\nEpoch: {epoch}")
        print(f"[Loss] Train: {train_loss:.4f} | Validation: {val_metrics['loss']:.4f} | Test: {test_metrics['loss']:.4f}")
        print(f"[Accuracy] Train Loss: {train_loss * 100:.2f}% | Validation: {val_metrics['acc'] * 100:.2f}% | Test: {test_metrics['acc'] * 100:.2f}%")
        print(f"[Precision] Validation: {val_metrics['precision'] * 100:.2f}% | Test: {test_metrics['precision'] * 100:.2f}%")
        print(f"[Recall] Validation: {val_metrics['recall'] * 100:.2f}% | Test: {test_metrics['recall'] * 100:.2f}%")
        print(f"[F1-Score] Validation: {val_metrics['f1'] * 100:.2f}% | Test: {test_metrics['f1'] * 100:.2f}%")

        # Optional: Generate classification report
        print("\nValidation Classification Report:")
        print(classification_report(
            y_true_val,
            y_pred_val,
            target_names=class_names,
            digits=4,
            zero_division=0
        ))

        print("\nTest Classification Report:")
        print(classification_report(
            y_true_test,
            y_pred_test,
            target_names=class_names,
            digits=4,
            zero_division=0
        ))

    print("------------------ Training and evaluation completed successfully ------------------\n\n")
    return model.history


def train_and_test_from_youtube(model, train_dataloader, test_dataloader, loss_fn, optimizer, accuracy_fn, epochs, device):
    print("------------------ Start train_and_test_from_youtube ------------------")

    model.to(device)

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch + 1}\n{'-' * 7}")

        # Training
        model.train()
        train_loss = 0
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)  # Move data to device

            # Forward pass
            y_pred = model(X)

            # Calculate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()  # Accumulate loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            if batch % 100 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

        train_loss /= len(train_dataloader)  # Average loss per batch

        # Testing
        model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)  # Move data to device

                # Forward pass
                test_pred = model(X)

                # Calculate loss and accuracy
                test_loss += loss_fn(test_pred, y).item()
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
                # test_acc += accuracy_fn(test_pred, y).item()
        test_loss /= len(test_dataloader)  # Average loss per batch
        test_acc /= len(test_dataloader)  # Average accuracy per batch

        # Print epoch metrics
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")
    print("------------------ train_and_test_from_youtube completed successfully ------------------")
    return model


# before fixing
def train_and_evaluate_firstVERSION_v1(epochs, train_loader, val_loader, test_loader, model, loss_function,
                                       optimizer, accuracy_metric, device, num_classes, class_names, debug=False):
    # Initialize metrics
    precision_metric = MulticlassPrecision(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    f1_score_metric = MulticlassF1Score(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    confusion_matrix_metric = ConfusionMatrix(num_classes=num_classes, task="multiclass").to(device)

    # Ensure the model has a history attribute
    if not hasattr(model, 'history'):
        model.history = {
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

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_precision = 0, 0, 0

        # Training Loop
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc += accuracy_metric(torch.argmax(y_pred, dim=1), y).item()
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

        # Validation Loop
        model.eval()
        val_loss, val_acc, val_precision, val_recall, val_f1 = 0, 0, 0, 0, 0
        with torch.inference_mode():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_pred = model(X)
                val_loss += loss_function(val_pred, y).item()

                val_acc += accuracy_metric(torch.argmax(val_pred, dim=1), y).item()
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

        # Testing Loop
        y_true = []
        y_pred = []
        test_loss, test_acc, test_precision, test_recall, test_f1 = 0, 0, 0, 0, 0
        with torch.inference_mode():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                test_pred = model(X)
                test_loss += loss_function(test_pred, y).item()

                # Compute metrics
                test_acc += accuracy_metric(torch.argmax(test_pred, dim=1), y).item()
                test_precision += precision_metric(test_pred, y).item()
                test_recall += recall_metric(test_pred, y).item()
                test_f1 += f1_score_metric(test_pred, y).item()
                confusion_matrix_metric.update(test_pred, y)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(test_pred.argmax(dim=1).cpu().numpy())

            # Average testing metrics over the entire loader
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
        # confusion_matrix_metric.reset()

        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

        print(f"\nEpoch: {epoch}")
        print(f"[Loss] Train: {train_loss:.4f} | Validation: {val_loss:.4f} | Test: {test_loss:.4f}")
        print(f"[Accuracy] Train: {train_acc * 100:.2f}% | Validation: {val_acc * 100:.2f}% | Test: {test_acc * 100:.2f}%")
        print(f"[Precision] Train: {train_precision * 100:.2f}% | Validation: {val_precision * 100:.2f}% | Test: {test_precision * 100:.2f}%")
        print(f"[Recall] Validation: {val_recall * 100:.2f}% | Test: {test_recall * 100:.2f}%")
        print(f"[F1-Score] Validation: {val_f1 * 100:.2f}% | Test: {test_f1 * 100:.2f}%")

    print("Training and evaluation completed successfully ------------------\n\n")
    return model.history


# Training, validation, and testing loop
def train_validate_test_with_torchmetrics(model, train_loader, num_classes, val_loader, test_loader, epochs, criterion, optimizer, device):
    """
    Function to train, validate, and test a PyTorch model using torchmetrics.

    Parameters:
    - model: The PyTorch model to be trained and evaluated.
    - train_loader, val_loader, test_loader: DataLoader objects for the training, validation, and test sets.
    - epochs: Number of training epochs.
    - criterion: Loss function.
    - optimizer: Optimizer.
    - device: Device for computation ("cpu" or "cuda").

    Returns:
    - model: The trained model.
    """
    model.to(device)

    # Initialize metrics
    train_precision = Precision(num_classes=num_classes, task="multiclass", average="weighted").to(device)
    val_precision = Precision(num_classes=num_classes, task="multiclass", average="weighted").to(device)
    val_recall = Recall(num_classes=num_classes, task="multiclass", average="weighted").to(device)
    val_f1 = F1Score(num_classes=num_classes, task="multiclass", average="weighted").to(device)
    test_precision = Precision(num_classes=num_classes, task="multiclass", average="weighted").to(device)
    test_recall = Recall(num_classes=num_classes, task="multiclass", average="weighted").to(device)
    test_f1 = F1Score(num_classes=num_classes, task="multiclass", average="weighted").to(device)
    test_cm = ConfusionMatrix(num_classes=num_classes, task="multiclass").to(device)  # Adjust `num_classes` as needed

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_total += targets.size(0)

            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == targets).sum().item()

            # Update training precision
            train_precision.update(predicted, targets)

        # Calculate training metrics
        train_acc = train_correct / train_total
        train_prec = train_precision.compute().item()

        model.record_metric("train_loss", train_loss / train_total)
        model.record_metric("train_acc", train_acc)
        model.record_metric("train_precision", train_prec)
        train_precision.reset()

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                val_total += targets.size(0)

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()

                # Update validation metrics
                val_precision.update(predicted, targets)
                val_recall.update(predicted, targets)
                val_f1.update(predicted, targets)

        # Calculate validation metrics
        val_acc = val_correct / val_total
        val_prec = val_precision.compute().item()
        val_rec = val_recall.compute().item()
        val_f1_score = val_f1.compute().item()

        model.record_metric("val_loss", val_loss / val_total)
        model.record_metric("val_acc", val_acc)
        model.record_metric("val_precision", val_prec)
        model.record_metric("val_recall", val_rec)
        model.record_metric("val_f1_score", val_f1_score)

        val_precision.reset()
        val_recall.reset()
        val_f1.reset()

        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Train Loss: {train_loss / train_total:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss / val_total:.4f}, Val Acc: {val_acc:.4f}")

    # Testing Phase
    test_loss, test_correct, test_total = 0.0, 0, 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            test_total += targets.size(0)

            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == targets).sum().item()

            # Update testing metrics
            test_precision.update(predicted, targets)
            test_recall.update(predicted, targets)
            test_f1.update(predicted, targets)
            test_cm.update(predicted, targets)

    # Calculate test metrics
    test_acc = test_correct / test_total
    test_prec = test_precision.compute().item()
    test_rec = test_recall.compute().item()
    test_f1_score = test_f1.compute().item()
    test_conf_matrix = test_cm.compute().cpu().numpy()

    model.record_metric("test_loss", test_loss / test_total)
    model.record_metric("test_acc", test_acc)
    model.record_metric("test_precision", test_prec)
    model.record_metric("test_recall", test_rec)
    model.record_metric("test_f1_score", test_f1_score)
    model.record_metric("confusion_matrix", test_conf_matrix.tolist())

    print(f"Test Results: "
          f"Loss: {test_loss / test_total:.4f}, Acc: {test_acc:.4f}, "
          f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1_score:.4f}")

    return model


class CNN_ModelV6_2(LightningModule):

    def __init__(self):
        super(CNN_ModelV6_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, len(class_names))


        # Initialize metrics
        self.train_precision = Precision(task="multiclass", num_classes=len(class_names))
        self.val_precision = Precision(task="multiclass", num_classes=len(class_names))
        self.val_recall = Recall(task="multiclass", num_classes=len(class_names))
        self.val_f1 = F1Score(task="multiclass", num_classes=len(class_names))
        self.test_precision = Precision(task="multiclass", num_classes=len(class_names))
        self.test_recall = Recall(task="multiclass", num_classes=len(class_names))
        self.test_f1 = F1Score(task="multiclass", num_classes=len(class_names))
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=len(class_names))

        # Initialize history dictionary
        self.history = {
            "train_loss": [], "train_acc": [], "train_precision": [],
            "val_loss": [], "val_acc": [], "val_precision": [],
            "val_recall": [], "val_f1_score": [],
            "test_loss": [], "test_acc": [], "test_precision": [],
            "test_recall": [], "test_f1_score": [],
            "confusion_matrix": []
        }

    def calculate_metrics(self, y_hat, y, stage='train'):
        pred = y_hat.argmax(dim=1)

        # Calculate basic metrics
        loss = F.cross_entropy(y_hat, y)
        acc = (pred == y).float().mean()

        # Calculate additional metrics based on stage
        if stage == 'train':
            precision = self.train_precision(pred, y)
            metrics = {
                'loss': loss,
                'acc': acc,
                'precision': precision
            }
        elif stage == 'val':
            precision = self.val_precision(pred, y)
            recall = self.val_recall(pred, y)
            f1 = self.val_f1(pred, y)
            conf_matrix = self.confusion_matrix(pred, y)
            metrics = {
                'loss': loss,
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': conf_matrix
            }
        else:  # test
            precision = self.test_precision(pred, y)
            recall = self.test_recall(pred, y)
            f1 = self.test_f1(pred, y)
            metrics = {
                'loss': loss,
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 54 * 54)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return F.log_softmax(X, dim=1)

    def on_train_epoch_start(self):
        self.train_precision.reset()

    def on_validation_epoch_start(self):
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.confusion_matrix.reset()

    def on_test_epoch_start(self):
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y_hat = self(X)
        metrics = self.calculate_metrics(y_hat, y, stage='train')

        # Log metrics
        self.log("train_loss", metrics['loss'])
        self.log("train_acc", metrics['acc'])
        self.log("train_precision", metrics['precision'])

        # Store in history
        self.history["train_loss"].append(metrics['loss'].item())
        self.history["train_acc"].append(metrics['acc'].item())
        self.history["train_precision"].append(metrics['precision'].item())

        return metrics['loss']

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        y_hat = self(X)
        metrics = self.calculate_metrics(y_hat, y, stage='val')

        # Log metrics
        self.log("val_loss", metrics['loss'])
        self.log("val_acc", metrics['acc'])
        self.log("val_precision", metrics['precision'])
        self.log("val_recall", metrics['recall'])
        self.log("val_f1_score", metrics['f1'])

        # Store in history
        self.history["val_loss"].append(metrics['loss'].item())
        self.history["val_acc"].append(metrics['acc'].item())
        self.history["val_precision"].append(metrics['precision'].item())
        self.history["val_recall"].append(metrics['recall'].item())
        self.history["val_f1_score"].append(metrics['f1'].item())

        if batch_idx == 0:  # Save confusion matrix once per epoch
            self.history["confusion_matrix"].append(metrics['confusion_matrix'].cpu().numpy())

    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        y_hat = self(X)
        metrics = self.calculate_metrics(y_hat, y, stage='test')

        # Log metrics
        self.log("test_loss", metrics['loss'])
        self.log("test_acc", metrics['acc'])
        self.log("test_precision", metrics['precision'])
        self.log("test_recall", metrics['recall'])
        self.log("test_f1_score", metrics['f1'])

        # Store in history
        self.history["test_loss"].append(metrics['loss'].item())
        self.history["test_acc"].append(metrics['acc'].item())
        self.history["test_precision"].append(metrics['precision'].item())
        self.history["test_recall"].append(metrics['recall'].item())
        self.history["test_f1_score"].append(metrics['f1'].item())


class CNN_ModelV6_1(LightningModule):

    def __init__(self):
        super(CNN_ModelV6_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, len(class_names))

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 54 * 54)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return F.log_softmax(X, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("test_loss", loss)
        self.log("test_acc", acc)