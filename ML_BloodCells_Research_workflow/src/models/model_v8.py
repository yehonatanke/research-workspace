import torch
from torch import nn
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from torchvision import models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from tqdm import tqdm


class ModelEffNetV2(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.base = models.efficientnet_v2_s(weights=weights)

        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False

        self.block = nn.Sequential(
            # nn.Linear(512, 128),
            nn.Linear(1280, 128),  # Changed from 512 to 1280
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.base.classifier = nn.Sequential()
        self.base.fc = nn.Sequential()

    def get_optimizer(self):
        return torch.optim.AdamW([
            {'params': self.base.parameters(), 'lr': 3e-5},
            {'params': self.block.parameters(), 'lr': 8e-4}
        ])

    def forward(self, x):
        x = self.base(x)
        x = self.block(x)
        return x


class TrainerModel8(nn.Module):
    def __init__(self, train_loader, val_loader, test_loader=None, num_classes=8, device='cpu'):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        self.model = ModelEffNetV2().to(self.device)
        self.optimizer = self.model.get_optimizer()
        self.loss_fxn = nn.CrossEntropyLoss()

        # Initialize all metrics
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.precision = Precision(task="multiclass", num_classes=num_classes).to(self.device)
        self.recall = Recall(task="multiclass", num_classes=num_classes).to(self.device)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes).to(self.device)
        self.confusion_matrix_metric = ConfusionMatrix(num_classes=num_classes, task="multiclass").to(self.device)

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_f1_score': [], 'val_f1_score': [],
            "test_loss": [], "test_acc": [],
            "test_precision": [], "test_recall": [],
            "test_f1_score": [], "confusion_matrix": []
        }

    def reset_metrics(self):
        """Reset all metrics"""
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.confusion_matrix_metric.reset()

    def training_step(self, x, y):
        pred = self.model(x)
        loss = self.loss_fxn(pred, y)

        # Calculate all metrics
        acc = self.accuracy(pred, y)
        prec = self.precision(pred, y)
        rec = self.recall(pred, y)
        f1 = self.f1(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, acc, prec, rec, f1

    def validation_step(self, x, y):
        with torch.inference_mode():
            pred = self.model(x)
            loss = self.loss_fxn(pred, y)

            # Calculate all metrics
            acc = self.accuracy(pred, y)
            prec = self.precision(pred, y)
            rec = self.recall(pred, y)
            f1 = self.f1(pred, y)

        return loss, acc, prec, rec, f1

    def process_batch(self, loader, step):
        step_name = step.__name__.replace('_', ' ').capitalize() if hasattr(step, '__name__') else "Processing"

        loss, acc, prec, rec, f1 = 0, 0, 0, 0, 0
        self.reset_metrics()
        for X, y in tqdm(loader, total=len(loader), desc=f"Processing Batch - {step_name}",
                         leave=False, position=2, bar_format="{l_bar}{bar} | Batch {n_fmt}/{total_fmt}"):
            X, y = X.to(self.device), y.to(self.device)
            l, a, p, r, f = step(X, y)
            loss += l.item()
            acc += a.item()
            prec += p.item()
            rec += r.item()
            f1 += f.item()

        n = len(loader)
        return loss / n, acc / n, prec / n, rec / n, f1 / n

    def train(self, epochs):
        for epoch in tqdm(range(epochs), desc="Overall Progress: Epochs", leave=True,
                          position=0, bar_format="{l_bar}{bar} | Batch {n_fmt}/{total_fmt}"):
            self.reset_metrics()
            # Training phase
            train_loss, train_acc, train_prec, train_rec, train_f1_score = self.process_batch(
                self.train_loader, self.training_step
            )

            # Validation phase
            val_loss, val_acc, val_prec, val_rec, val_f1_score = self.process_batch(
                self.val_loader, self.validation_step
            )

            # Update history
            metrics = [
                train_loss, val_loss,
                train_acc, val_acc,
                train_prec, val_prec,
                train_rec, val_rec,
                train_f1_score, val_f1_score
            ]

            for item, value in zip(self.history.keys(), metrics):
                self.history[item].append(value)

            print(
                f"[Epoch: {epoch + 1}] "
                f"Train: [loss: {train_loss:.3f} acc: {train_acc:.3f} "
                f"prec: {train_prec:.3f} rec: {train_rec:.3f} f1: {train_f1_score:.3f}] "
                f"Val: [loss: {val_loss:.3f} acc: {val_acc:.3f} "
                f"prec: {val_prec:.3f} rec: {val_rec:.3f} f1: {val_f1_score:.3f}]"
            )
            print(f"\nEpoch {epoch + 1}/{epochs} Performance Report:")
            print(f"└─ [Train] Loss: {train_loss:.4f} | Accuracy: {train_acc * 100:.2f}% | Precision: {train_prec:.2f}")
            print(f"└─ [Validation] Loss: {val_loss:.4f} | Accuracy: {val_acc * 100:.2f}% | Precision: {val_prec:.2f} | Recall: {val_rec:.2f} | F1-Score: {val_f1_score:.2f}")
        print("Finished training and validation.")

    def test(self):
        """
        Evaluate the model on the test set after training is complete.
        Returns a dictionary with test metrics.
        """
        if self.test_loader is None:
            raise ValueError("Test loader was not provided during initialization")

        self.model.eval()
        self.reset_metrics()

        test_loss = 0
        with torch.inference_mode():
            for X, y in tqdm(self.test_loader, desc=f"Testing Phase",
                             leave=False, position=2, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}"):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fxn(pred, y)
                test_loss += loss.item()

                # Calculate all metrics
                self.accuracy(pred, y)
                self.precision(pred, y)
                self.recall(pred, y)
                self.f1(pred, y)
                self.confusion_matrix_metric(pred, y)

        # Calculate average test loss
        test_loss /= len(self.test_loader)

        # Compute final metrics
        test_acc = self.accuracy.compute()
        test_precision = self.precision.compute()
        test_recall = self.recall.compute()
        test_f1 = self.f1.compute()
        confusion_matrix = self.confusion_matrix_metric.compute()

        self.history["test_loss"].append(test_loss)
        self.history["test_acc"].append(test_acc.item())
        self.history["test_precision"].append(test_precision.item())
        self.history["test_recall"].append(test_recall.item())
        self.history["test_f1_score"].append(test_f1.item())
        self.history["confusion_matrix"].append(confusion_matrix.cpu().numpy())
        self.confusion_matrix_metric.reset()
        # Print test results
        print(f"\n[Test] Loss: {test_loss:.4f} | Accuracy: {test_acc * 100:.2f} | Precision: {test_precision:.2f} | Recall: {test_recall:.2f} | F1-Score: {test_f1:.2f}")
        print("Finished test evaluation.")
