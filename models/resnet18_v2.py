class ModelResNet18(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.base = models.resnet18(weights=weights)

        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False

        self.block = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.base.classifier = nn.Sequential()
        self.base.fc = nn.Sequential()

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

    def get_optimizer(self):
        return torch.optim.AdamW([
            {'params': self.base.parameters(), 'lr': 3e-5},
            {'params': self.block.parameters(), 'lr': 8e-4}
        ])

    def forward(self, x):
        x = self.base(x)
        x = self.block(x)
        return x

    def record_metric(self, metric_name: str, value: float):
        if metric_name not in self.history:
            self.history[metric_name] = []
        self.history[metric_name].append(value)

    def get_history(self, metric_name: str):
        return self.history.get(metric_name, [])

    def get_all_metrics(self):
        return self.history
