import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerInverseAutoencoder(nn.Module):
    """Fully connected autoencoder with inverse constraint."""
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.encoder_activations = []
        self.decoder_activations = []
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.encoder_layers = nn.ModuleList(encoder_layers)
        decoder_layers = []
        reversed_dims = hidden_dims[::-1]
        for i, hidden_dim in enumerate(reversed_dims[1:] + [input_dim]):
            decoder_layers.append(nn.Linear(reversed_dims[i], hidden_dim))
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.activation_fn = self._get_activation_fn()
    def _get_activation_fn(self):
        if self.activation == 'relu':
            return F.relu
        elif self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'sigmoid':
            return torch.sigmoid
        else:
            return F.relu
    def encode(self, x, store_activations=False):
        if store_activations:
            self.encoder_activations = []
        h = x.view(x.size(0), -1)
        for i, layer in enumerate(self.encoder_layers):
            if store_activations:
                self.encoder_activations.append(h.clone())
            h = layer(h)
            if i < len(self.encoder_layers) - 1:
                h = self.activation_fn(h)
        if store_activations:
            self.encoder_activations.append(h.clone())
        return h
    def decode(self, z, store_activations=False):
        if store_activations:
            self.decoder_activations = []
        h = z
        for i, layer in enumerate(self.decoder_layers):
            if store_activations:
                self.decoder_activations.append(h.clone())
            h = layer(h)
            if i < len(self.decoder_layers) - 1:
                h = self.activation_fn(h)
        if store_activations:
            self.decoder_activations.append(h.clone())
        return h.view(h.size(0), 1, 28, 28)
    def forward(self, x, store_activations=False):
        if store_activations:
            self.encoder_activations = []
            self.decoder_activations = []
        encoded = self.encode(x, store_activations)
        decoded = self.decode(encoded, store_activations)
        return decoded, encoded

class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder with inverse constraint."""
    def __init__(self, channels=[32, 64, 128]):
        super().__init__()
        self.channels = channels
        self.encoder_activations = []
        self.decoder_activations = []
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels[0], 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[1], 4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels[1], channels[0], 4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels[0], 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def encode(self, x, store_activations=False):
        if store_activations:
            self.encoder_activations = []
        h = x
        for i, layer in enumerate(self.encoder):
            if store_activations and isinstance(layer, nn.Conv2d):
                self.encoder_activations.append(h.clone())
            h = layer(h)
        if store_activations:
            self.encoder_activations.append(h.clone())
        return h
    def decode(self, z, store_activations=False, debug=False):
        if store_activations:
            self.decoder_activations = []
        h = z
        for i, layer in enumerate(self.decoder):
            if store_activations and isinstance(layer, nn.ConvTranspose2d):
                self.decoder_activations.append(h.clone())
            h = layer(h)
        if store_activations:
            self.decoder_activations.append(h.clone())
        if h.shape[-1] != 28 or h.shape[-2] != 28:
            if debug:
                print(f"Warning: Decoder output shape {h.shape} is not 28x28. Interpolating.")
            h = F.interpolate(h, size=(28, 28), mode='bilinear', align_corners=False)
        return h
    def forward(self, x, store_activations=False):
        if store_activations:
            self.encoder_activations = []
            self.decoder_activations = []
        encoded = self.encode(x, store_activations)
        decoded = self.decode(encoded, store_activations)
        return decoded, encoded
