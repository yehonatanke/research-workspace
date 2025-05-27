import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calculate_accuracy(predictions, targets, idx_to_vocab):
    pred_indices = torch.argmax(predictions, dim=-1)
    pad_idx = 0
    mask = (targets != pad_idx)
    correct = (pred_indices == targets) & mask
    total = mask.sum()
    if total == 0:
        return 0.0
    return (correct.sum().float() / total.float()).item()

class DenoisingAutoEncoder(nn.Module):
    """
    Denoising AutoEncoder for text.
    Encoder: Embedding + LSTM + Linear
    Decoder: Linear + LSTM + Output projection
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1):
        super(DenoisingAutoEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                                   batch_first=True, dropout=0.1)
        self.encoder_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.decoder_fc = nn.Linear(hidden_dim // 2, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                                   batch_first=True, dropout=0.1)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
    def forward(self, noisy_input):
        batch_size, seq_len = noisy_input.shape
        embedded = self.embedding(noisy_input)
        encoder_out, (hidden, cell) = self.encoder_lstm(embedded)
        encoded = self.encoder_fc(encoder_out[:, -1, :])
        encoded = self.dropout(encoded)
        decoded_hidden = self.decoder_fc(encoded)
        decoded_hidden = self.dropout(decoded_hidden)
        decoder_input = decoded_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_out, _ = self.decoder_lstm(decoder_input)
        output = self.output_projection(decoder_out)
        return output

def train_model(model, train_loader, val_loader, idx_to_vocab, num_epochs=10, lr=0.001, show_progress=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        num_train_batches = 0
        for batch in train_loader:
            noisy_input = batch['noisy'].to(device)
            clean_target = batch['clean'].to(device)
            optimizer.zero_grad()
            outputs = model(noisy_input)
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = clean_target.view(-1)
            loss = criterion(outputs_flat, targets_flat)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += calculate_accuracy(outputs, clean_target, idx_to_vocab)
            num_train_batches += 1
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                noisy_input = batch['noisy'].to(device)
                clean_target = batch['clean'].to(device)
                outputs = model(noisy_input)
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = clean_target.view(-1)
                loss = criterion(outputs_flat, targets_flat)
                val_loss += loss.item()
                val_acc += calculate_accuracy(outputs, clean_target, idx_to_vocab)
                num_val_batches += 1
        avg_train_loss = train_loss / num_train_batches
        avg_val_loss = val_loss / num_val_batches
        avg_train_acc = train_acc / num_train_batches
        avg_val_acc = val_acc / num_val_batches
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(avg_train_acc)
        val_accuracies.append(avg_val_acc)
        if show_progress and epoch % 10 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}]  [Train Loss: {avg_train_loss:.4f}, Train Acc: {100 * avg_train_acc:.2f}%]  [Val Loss: {avg_val_loss:.4f}, Val Acc: {100 * avg_val_acc:.2f}%]")
    return train_losses, val_losses, train_accuracies, val_accuracies

def test_model(model, test_loader, idx_to_vocab):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    test_loss = 0.0
    test_acc = 0.0
    num_batches = 0
    predictions_text = []
    actuals_text = []
    noisy_text = []
    with torch.no_grad():
        for batch in test_loader:
            noisy_input = batch['noisy'].to(device)
            clean_target = batch['clean'].to(device)
            outputs = model(noisy_input)
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = clean_target.view(-1)
            loss = criterion(outputs_flat, targets_flat)
            test_loss += loss.item()
            test_acc += calculate_accuracy(outputs, clean_target, idx_to_vocab)
            num_batches += 1
            if len(predictions_text) < 10:
                pred_indices = torch.argmax(outputs, dim=-1)
                for i in range(min(5, pred_indices.shape[0])):
                    pred_words = [idx_to_vocab.get(idx.item(), '<UNK>')
                                 for idx in pred_indices[i] if idx.item() != 0]
                    target_words = [idx_to_vocab.get(idx.item(), '<UNK>')
                                   for idx in clean_target[i] if idx.item() != 0]
                    predictions_text.append(' '.join(pred_words))
                    actuals_text.append(' '.join(target_words))
                    noisy_text.append(batch['noisy_text'][i])
    avg_test_loss = test_loss / num_batches
    avg_test_acc = test_acc / num_batches
    print(f"\nTest Results:")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {100 * avg_test_acc:.2f}%")
    print(f"\nSample Predictions \n")
    for i in range(min(5, len(predictions_text))):
        print(f"Example {i+1}:")
        print(f"Noisy:     {noisy_text[i]}")
        print(f"Predicted: {predictions_text[i]}")
        print(f"Actual:    {actuals_text[i]}")
        print("-" * 60)
    return avg_test_loss, avg_test_acc 