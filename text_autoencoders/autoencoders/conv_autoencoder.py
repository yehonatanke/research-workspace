import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_device
from collections import defaultdict

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder with inverse constraint.
    """
    def __init__(self, channels=[32, 64, 128]):
        super(ConvAutoencoder, self).__init__()
        self.channels = channels
        self.encoder_activations = []
        self.decoder_activations = []
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels[0], 4, stride=2, padding=1),  # 28->14
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], 4, stride=2, padding=1),  # 14->7
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], 4, stride=2, padding=1),  # 7->3
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[1], 4, stride=2, padding=1, output_padding=1),  # 3->7
            nn.ReLU(),
            nn.ConvTranspose2d(channels[1], channels[0], 4, stride=2, padding=1, output_padding=1),  # 7->14
            nn.ReLU(),
            nn.ConvTranspose2d(channels[0], 1, 4, stride=2, padding=1),  # 14->28
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

def run_conv_model(train_loader, epochs=30, lambda_inverse=0.05, debug=False):
    print("Running Convolutional Autoencoder...")
    if debug:
        print("Testing convolutional network dimensions...")
        test_model = test_conv_dimensions()
    model = ConvAutoencoder(channels=[32, 64, 128])
    losses = train_model(model, train_loader, num_epochs=epochs, lambda_inverse=lambda_inverse)
    return model, losses

def run_conv_models_multiple(architectures, train_loader, epochs=50, plot_res=False, debug=False):
    results = {}
    print("Running Convolutional Autoencoder...")
    for arch_name, params in architectures.items():
        print(f"\nTraining {arch_name} architecture...")
        if debug:
            print("Testing convolutional network dimensions...")
            test_model = test_conv_dimensions()
        model = ConvAutoencoder(channels=params['channels'])
        model.to(get_device())
        losses = train_model(model, train_loader, num_epochs=epochs, lambda_inverse=params['lambda_inverse'])
        results[arch_name] = {
            'model': model,
            'losses': losses,
        }
    best_arch = min(results.keys(), key=lambda x: results[x]['losses']['total'][-1])
    worst_arch = max(results.keys(), key=lambda x: results[x]['losses']['total'][-1])
    print(f"\nBest convolutional architecture: {best_arch}")
    print(f"Worst convolutional architecture: {worst_arch}")
    return results, best_arch, worst_arch

def test_conv_dimensions():
    model = ConvAutoencoder(channels=[32, 64, 128])
    model.eval()
    sample_input = torch.randn(1, 1, 28, 28)
    print(f"Input shape: {sample_input.shape}")
    encoded = model.encode(sample_input)
    print(f"Encoded shape: {encoded.shape}")
    decoded = model.decode(encoded)
    print(f"Decoded shape: {decoded.shape}")
    reconstructed, _ = model(sample_input)
    print(f"Reconstructed shape: {reconstructed.shape}")
    return model

def train_model(model, train_loader, num_epochs=50, learning_rate=1e-3, lambda_inverse=0.1):
    device = get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = defaultdict(list)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_inverse_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            total_loss, recon_loss, inv_loss = inverse_loss(model, data, lambda_inverse)
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_inverse_loss += inv_loss.item()
        epoch_loss /= len(train_loader)
        epoch_recon_loss /= len(train_loader)
        epoch_inverse_loss /= len(train_loader)
        losses['total'].append(epoch_loss)
        losses['reconstruction'].append(epoch_recon_loss)
        losses['inverse'].append(epoch_inverse_loss)
        if epoch % 10 == 0:
            print(f'[Epoch {epoch}/{num_epochs}] [Total Loss: {epoch_loss:.4f}, '
                  f'Recon Loss: {epoch_recon_loss:.4f}, Inverse Loss: {epoch_inverse_loss:.4f}]')
    return losses

def inverse_loss(model, x, lambda_inverse=1.0):
    reconstructed, encoded = model(x, store_activations=True)
    recon_loss = F.mse_loss(reconstructed, x)
    inverse_loss_val = torch.tensor(0.0).to(x.device)
    if hasattr(model, 'encoder_activations') and hasattr(model, 'decoder_activations'):
        if len(model.encoder_activations) > 0 and len(model.decoder_activations) > 0:
            min_len = min(len(model.encoder_activations), len(model.decoder_activations))
            for i in range(min_len):
                enc_act = model.encoder_activations[i]
                dec_act = model.decoder_activations[min_len - 1 - i]
                if enc_act.shape != dec_act.shape:
                    if len(enc_act.shape) == 4 and len(dec_act.shape) == 4:
                        dec_act = F.interpolate(dec_act, size=enc_act.shape[2:], mode='bilinear', align_corners=False)
                if enc_act.shape == dec_act.shape:
                    inverse_loss_val += F.mse_loss(dec_act, enc_act)
    total_loss = recon_loss + lambda_inverse * inverse_loss_val
    return total_loss, recon_loss, inverse_loss_val 