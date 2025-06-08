import torch
import torch.nn as nn

class DenoisingAutoEncoder(nn.Module):
    """Denoising autoencoder for text (embedding + LSTM)."""
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.encoder_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.decoder_fc = nn.Linear(hidden_dim // 2, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
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

class EmotionClassifier(nn.Module):
    """Classifier using pre-trained encoder (transfer learning)."""
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
