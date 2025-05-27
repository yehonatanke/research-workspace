import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, confusion_matrix

def create_baseline_classifier(vocab_size, num_classes=2):
    class BaselineClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.1)
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        def forward(self, input_ids):
            embedded = self.embedding(input_ids)
            lstm_out, (hidden, _) = self.lstm(embedded)
            logits = self.classifier(lstm_out[:, -1, :])
            return logits
    return BaselineClassifier(vocab_size, num_classes=num_classes)

class EmotionClassifier(nn.Module):
    def __init__(self, pretrained_encoder, num_classes=2, freeze_encoder=True, dropout=0.3):
        super().__init__()
        self.embedding = pretrained_encoder.embedding
        self.encoder_lstm = pretrained_encoder.encoder_lstm
        self.encoder_fc = pretrained_encoder.encoder_fc
        if freeze_encoder:
            for param in self.embedding.parameters():
                param.requires_grad = False
            for param in self.encoder_lstm.parameters():
                param.requires_grad = False
            for param in self.encoder_fc.parameters():
                param.requires_grad = False
        encoder_output_dim = pretrained_encoder.encoder_fc.out_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_output_dim, encoder_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_output_dim // 2, num_classes)
        )
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        encoder_out, (hidden, cell) = self.encoder_lstm(embedded)
        encoded = self.encoder_fc(encoder_out[:, -1, :])
        logits = self.classifier(encoded)
        return logits

def train_emotion_classifier(model, train_loader, val_loader, num_epochs=15, lr=0.001, show_progress=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        if show_progress and epoch % 10 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}]  [Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {100 * train_acc:.2f}%]  [Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100 * val_acc:.2f}%]")
    return train_losses, val_losses, train_accuracies, val_accuracies

def test_emotion_classifier(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_acc = test_correct / test_total
    print(f"\nClassifier Results:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Correct Predictions: {test_correct}/{test_total}")
    print(f"\nClassification Report")
    print(classification_report(all_labels, all_predictions, target_names=['Negative', 'Positive'], zero_division=0))
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"\nConfusion Matrix")
    print(f"Confusion Matrix:")
    print(f"              Predicted")
    print(f"Actual    Neg    Pos")
    print(f"Neg      {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"Pos      {cm[1,0]:4d}   {cm[1,1]:4d}")
    return test_acc, all_predictions, all_labels 