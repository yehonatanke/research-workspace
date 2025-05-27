import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from .utils import get_device

class LayerInverseAutoencoder(nn.Module):
    """
    Autoencoder with constraint that decoder layers approximate inverse of encoder layers.
    """
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], activation='relu'):
        super(LayerInverseAutoencoder, self).__init__()
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

def inverse_loss(model, x, lambda_inverse=1.0):
    reconstructed, encoded = model(x, store_activations=True)
    recon_loss = F.mse_loss(reconstructed, x)
    inverse_loss_val = torch.tensor(0.0).to(x.device)
    if hasattr(model, 'encoder_activations') and hasattr(model, 'decoder_activations'):
        if len(model.encoder_activations) > 0 and len(model.decoder_activations) > 0:
            if isinstance(model, LayerInverseAutoencoder):
                h_decoded = encoded.clone()
                decoder_outputs_at_layers = []
                for i, layer in enumerate(model.decoder_layers):
                    h_decoded = layer(h_decoded)
                    if i < len(model.decoder_layers) - 1:
                        h_decoded = model.activation_fn(h_decoded)
                    decoder_outputs_at_layers.append(h_decoded.clone())
                min_len = min(len(model.encoder_activations), len(decoder_outputs_at_layers))
                for i in range(min_len):
                    enc_act = model.encoder_activations[i]
                    dec_output = decoder_outputs_at_layers[min_len - 1 - i]
                    if enc_act.shape == dec_output.shape:
                        inverse_loss_val += F.mse_loss(dec_output, enc_act)
    total_loss = recon_loss + lambda_inverse * inverse_loss_val
    return total_loss, recon_loss, inverse_loss_val

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

def evaluate_layer_inverseness(model, test_loader):
    device = get_device()
    model.eval()
    layer_similarities = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            _, _ = model(data, store_activations=True)
            if hasattr(model, 'encoder_activations') and hasattr(model, 'decoder_activations'):
                if len(model.encoder_activations) > 0 and len(model.decoder_activations) > 0:
                    min_layers = min(len(model.encoder_activations), len(model.decoder_activations))
                    for i in range(min_layers):
                        enc_act = model.encoder_activations[i]
                        dec_act = model.decoder_activations[-(i+1)]
                        if enc_act.shape == dec_act.shape:
                            similarity = F.mse_loss(dec_act, enc_act).item()
                            if len(layer_similarities) <= i:
                                layer_similarities.append([])
                            layer_similarities[i].append(similarity)
            break
    avg_similarities = [np.mean(sim_list) for sim_list in layer_similarities]
    return avg_similarities

def plot_results(model, test_loader, num_samples=8):
    device = get_device()
    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstructed, _ = model(data)
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, num_samples, figsize=(12, 4))
            for i in range(num_samples):
                axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                axes[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')
            plt.tight_layout()
            plt.show()
            break

def run_fc_models(architectures, train_loader, test_loader, epochs, plot_res=False):
    results = {}
    print("Running Fully Connected Autoencoder...\n")
    for arch_name, params in architectures.items():
        print(f"\nTraining {arch_name} architecture...")
        model = LayerInverseAutoencoder(
            input_dim=784,
            hidden_dims=params['hidden_dims'],
            activation='relu'
        )
        losses = train_model(model, train_loader, num_epochs=epochs,
                           lambda_inverse=params['lambda_inverse'])
        similarities = evaluate_layer_inverseness(model, test_loader)
        results[arch_name] = {
            'model': model,
            'losses': losses,
            'layer_similarities': similarities
        }
        print(f"Layer similarities for {arch_name}: {similarities}")
    best_arch = min(results.keys(), key=lambda x: results[x]['losses']['total'][-1])
    worst_arch = max(results.keys(), key=lambda x: results[x]['losses']['total'][-1])
    print(f"\nBest architecture: {best_arch}")
    print(f"Worst architecture: {worst_arch}")
    if plot_res:
        print("\nResults for best model:")
        plot_results(results[best_arch]['model'], test_loader)
        print("\nResults for worst model:")
        plot_results(results[worst_arch]['model'], test_loader)
    return results, best_arch, worst_arch 