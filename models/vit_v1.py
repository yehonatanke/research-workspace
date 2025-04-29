from torch.optim import lr_scheduler
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTClassifierV1:
    def __init__(self, num_classes=8, use_pretrained=True, device=None):
        """
        Initialize the BloodCellViTClassifier.

        Args:
            num_classes (int): Number of output classes
            use_pretrained (bool): Whether to use pretrained weights
            device (torch.device): Device to run the model on
        """
        self.num_classes = num_classes
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the Vision Transformer model
        try:
            # Newer PyTorch versions
            if use_pretrained:
                self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            else:
                self.model = vit_b_16(weights=None)
        except:
            # Older PyTorch versions
            try:
                self.model = vit_b_16(pretrained=use_pretrained)
            except:
                # Fallback for any other implementation
                self.model = vit_b_16()
                if use_pretrained:
                    print("Warning: Pretrained weights not loaded. Check PyTorch version.")

        # Modify the classification head for the number of classes
        # Check the structure of the head to handle different PyTorch versions
        if hasattr(self.model, 'heads'):
            if isinstance(self.model.heads, nn.Linear):
                # If heads is a single Linear layer
                in_features = self.model.heads.in_features
                self.model.heads = nn.Linear(in_features, num_classes)
            elif isinstance(self.model.heads, nn.Sequential):
                # If heads is a Sequential container
                for i, layer in enumerate(self.model.heads):
                    if isinstance(layer, nn.Linear):
                        in_features = layer.in_features
                        self.model.heads[i] = nn.Linear(in_features, num_classes)
                        break
        elif hasattr(self.model, 'head'):
            # Some implementations use 'head' instead of 'heads'
            if isinstance(self.model.head, nn.Linear):
                in_features = self.model.head.in_features
                self.model.head = nn.Linear(in_features, num_classes)

        # Move the model to the specified device
        self.model = self.model.to(self.device)

    def train_model(self, train_loader, val_loader, criterion=None, optimizer=None, scheduler=None,
                    num_epochs=25, save_path=None):
        """
        Train the model on the provided data loaders.

        Args:
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: Adam)
            scheduler: Learning rate scheduler (default: None)
            num_epochs: Number of training epochs
            save_path: Path to save the best model weights

        Returns:
            model: Best performing model
            history: Training history
        """
        since = time.time()

        # Default criterion and optimizer if not provided
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        if optimizer is None:
            optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)

        if scheduler is None:
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Create directory for saving models if it doesn't exist
        if save_path and not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        # For tracking best model
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        # For storing training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                    dataloader = train_loader
                else:
                    self.model.eval()   # Set model to evaluate mode
                    dataloader = val_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs} - {phase}')
                for inputs, labels in progress_bar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # Update progress bar
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

                if phase == 'train' and scheduler is not None:
                    scheduler.step()

                # Calculate epoch statistics
                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)

                # Store statistics for later plotting
                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc.item())
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc.item())

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Deep copy the model if best performance
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                        print(f"Model saved to {save_path}")

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')

        # Load best model weights
        self.model.load_state_dict(best_model_wts)

        return self.model, history

    def evaluate(self, test_loader):
        """
        Evaluate the model on a test set.

        Args:
            test_loader: DataLoader for test set

        Returns:
            test_loss: Average loss on test set
            test_acc: Accuracy on test set
            predictions: List of predictions
            true_labels: List of true labels
            class_accuracies: Per-class accuracy dict
        """
        self.model.eval()

        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()

        # Lists to store predictions and true labels for further analysis
        all_predictions = []
        all_labels = []

        # Dictionary to store per-class accuracy
        class_correct = {i: 0 for i in range(self.num_classes)}
        class_total = {i: 0 for i in range(self.num_classes)}

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Evaluating'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()

                # Store predictions and true labels
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if preds[i] == labels[i]:
                        class_correct[label] += 1

        # Calculate average test loss and accuracy
        test_loss = test_loss / len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)

        # Calculate per-class accuracy
        class_accuracies = {cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
                          for cls in class_total.keys()}

        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print('\nPer-class accuracies:')
        for cls in sorted(class_accuracies.keys()):
            print(f'Class {cls}: {class_accuracies[cls]:.4f}')

        return test_loss, test_acc, all_predictions, all_labels, class_accuracies

    def predict(self, inputs):
        """
        Make predictions on new inputs.

        Args:
            inputs: Input tensor or batch of tensors

        Returns:
            predictions: Model predictions
            probabilities: Prediction probabilities
        """
        self.model.eval()

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs)

        if inputs.dim() == 3:  # Single image
            inputs = inputs.unsqueeze(0)  # Add batch dimension

        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

        return predictions.cpu().numpy(), probabilities.cpu().numpy()

    def save_model(self, path):
        """Save the model to the specified path."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load the model from the specified path."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


def run_training(train_loader, val_loader, test_loader=None,
                num_classes=8, num_epochs=25, use_pretrained=True,
                learning_rate=0.001, weight_decay=0.01):
    """
    Run the full training pipeline with default parameters.

    Args:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set (optional)
        num_classes: Number of classes
        num_epochs: Number of training epochs
        use_pretrained: Whether to use pretrained weights
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer

    Returns:
        model: Trained model
        history: Training history
        test_results: Test results (if test_loader is provided)
    """
    # Initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    classifier = BloodCellViTClassifier(
        num_classes=num_classes,
        use_pretrained=use_pretrained,
        device=device
    )

    # Define optimizer and loss
    optimizer = optim.AdamW(classifier.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Use CosineAnnealingLR scheduler with warmup
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Train the model
    model, history = classifier.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        save_path="models/bloodcell_vit_best.pth"
    )

    # Test the model if test_loader is provided
    test_results = None
    if test_loader:
        print("\nEvaluating on test set:")
        test_results = classifier.evaluate(test_loader)

    return classifier, history, test_results
