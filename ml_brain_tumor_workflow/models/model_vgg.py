import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import shutil
from typing import Tuple, Dict, List


class ImageDataset(Dataset):
    def __init__(self, dir_path: str, img_size: Tuple[int, int] = (224, 224), transform=None):
        self.dir_path = dir_path
        self.img_size = img_size
        self.transform = transform
        self.images, self.labels, self.label_dict = self._load_data()

    def _load_data(self) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
        """Load and preprocess images from directory."""
        X, y = [], []
        labels = {}
        
        for i, path in enumerate(sorted(os.listdir(self.dir_path))):
            if path.startswith('.'):
                continue
                
            labels[i] = path
            class_path = os.path.join(self.dir_path, path)
            
            for file in os.listdir(class_path):
                if file.startswith('.'):
                    continue
                    
                img_path = os.path.join(class_path, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, self.img_size)
                X.append(img)
                y.append(i)
                
        print(f'{len(X)} images loaded from {self.dir_path} directory.')
        return X, y, labels
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class VGGClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(VGGClassifier, self).__init__()
        self.vgg = models.vgg16(pretrained=pretrained)
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.vgg.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg(x)


def plot_confusion_matrix(cm: np.ndarray,
                        classes: List[Tuple[int, str]],
                        normalize: bool = False,
                        title: str = 'Confusion matrix',
                        cmap: plt.cm = plt.cm.Blues) -> None:
    """Plot confusion matrix with optional normalization."""
    plt.figure(figsize=(6, 6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, [c[1] for c in classes], rotation=90)
    plt.yticks(tick_marks, [c[1] for c in classes])
    
    thresh = cm.max() / 2.
    cm = np.round(cm, 2)
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
                
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                num_epochs: int,
                device: torch.device) -> Dict:
    """Train the model and return training history."""
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_correct / val_total)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {history["train_loss"][-1]:.4f}, Train Acc: {history["train_acc"][-1]:.4f}')
        print(f'Val Loss: {history["val_loss"][-1]:.4f}, Val Acc: {history["val_acc"][-1]:.4f}')
        
    return history


def plot_training_history(history: Dict) -> None:
    """Plot training and validation metrics."""
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Train')
    plt.plot(epochs_range, history['val_acc'], label='Validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Train')
    plt.plot(epochs_range, history['val_loss'], label='Validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    
    plt.tight_layout()
    plt.show()

def run_model_vgg():
    TRAIN_DIR = 'TRAIN/'
    VAL_DIR = 'VAL/'
    TEST_DIR = 'TEST/'
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageDataset(TRAIN_DIR, IMG_SIZE, transform)
    val_dataset = ImageDataset(VAL_DIR, IMG_SIZE, transform)
    test_dataset = ImageDataset(TEST_DIR, IMG_SIZE, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    num_classes = len(train_dataset.label_dict)
    model = VGGClassifier(num_classes).to(DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
    
    history = train_model(model, train_loader, val_loader, criterion,
                         optimizer, NUM_EPOCHS, DEVICE)
    
    plot_training_history(history)
    
    model.eval()
    test_predictions = []
    test_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predicted = (outputs > 0.5).float().cpu().numpy()
            test_predictions.extend(predicted)
            test_labels.extend(labels.numpy())
    
    confusion_mtx = confusion_matrix(test_labels, test_predictions)
    plot_confusion_matrix(confusion_mtx,
                         classes=list(train_dataset.label_dict.items()),
                         normalize=False,
                         title='Test Set Confusion Matrix')
    
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f'Test Accuracy: {test_accuracy:.4f}')

