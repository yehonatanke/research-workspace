import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch.nn.functional as F

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def plot_loss_history_v1(results_dict, title="Training Loss"):
    plt.figure(figsize=(10, 6))
    for name, result in results_dict.items():
        if 'losses' in result and 'total' in result['losses']:
            epochs = range(len(result['losses']['total']))
            plt.plot(epochs, result['losses']['total'], label=f'{name} - Total Loss')
            plt.plot(epochs, result['losses']['reconstruction'], label=f'{name} - Recon Loss')
            plt.plot(epochs, result['losses']['inverse'], label=f'{name} - Inverse Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss_history(results_dict, title="Training Loss"):
    palette = sns.color_palette("Paired", 8)
    selected_colors = [palette[1], palette[3], palette[7]]
    model_names = list(results_dict.keys())
    model_colors = {name: selected_colors[i] for i, name in enumerate(model_names)}
    all_data = []
    for name, result in results_dict.items():
        if 'losses' in result:
            losses = result['losses']
            for epoch, (t, r, inv) in enumerate(zip(losses['total'], losses['reconstruction'], losses['inverse'])):
                all_data.append({'Epoch': epoch, 'Loss': t, 'Component': 'Total', 'Model': name})
                all_data.append({'Epoch': epoch, 'Loss': r, 'Component': 'Reconstruction', 'Model': name})
                all_data.append({'Epoch': epoch, 'Loss': inv, 'Component': 'Inverse', 'Model': name})
    df = pd.DataFrame(all_data)
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    for model_name, model_df in df.groupby("Model"):
        base_color = model_colors[model_name]
        for component, linestyle in zip(["Total", "Reconstruction", "Inverse"], ["-", "--", ":"]):
            comp_df = model_df[model_df["Component"] == component]
            sns.lineplot(
                data=comp_df,
                x="Epoch",
                y="Loss",
                label=f"{model_name} - {component}",
                color=base_color,
                linestyle=linestyle,
            )
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_conv_loss_history(losses, title="Convolutional Autoencoder Loss"):
    epochs = range(len(losses['total']))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses['total'], label='Total Loss')
    plt.plot(epochs, losses['reconstruction'], label='Reconstruction Loss')
    plt.plot(epochs, losses['inverse'], label='Inverse Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_decode_results(model, test_loader, num_samples=8, noise_std=0.1):
    device = get_device()
    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstructed, encoded = model(data)
            noise = torch.randn_like(data) * noise_std
            noisy_data = data + noise
            noisy_reconstructed, noisy_encoded = model(noisy_data)
            pure_noise = torch.randn_like(data) * noise_std
            noise_reconstructed, noise_encoded = model(pure_noise)
            decoded_clean_encoding = model.decode(encoded) if hasattr(model, 'decode') else reconstructed
            decoded_noisy_encoding = model.decode(noisy_encoded) if hasattr(model, 'decode') else noisy_reconstructed
            decoded_noise_encoding = model.decode(noise_encoded) if hasattr(model, 'decode') else noise_reconstructed
            fig, axes = plt.subplots(4, num_samples, figsize=(16, 8))
            for i in range(num_samples):
                axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[0, i].set_title('Original Image', fontsize=10)
                axes[0, i].axis('off')
                axes[1, i].imshow(decoded_clean_encoding[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[1, i].set_title('Decoded Clean Encoding', fontsize=10)
                axes[1, i].axis('off')
                axes[2, i].imshow(decoded_noisy_encoding[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[2, i].set_title('Decoded Noisy Encoding', fontsize=10)
                axes[2, i].axis('off')
                axes[3, i].imshow(decoded_noise_encoding[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[3, i].set_title('Decoded Pure Noise', fontsize=10)
                axes[3, i].axis('off')
            plt.tight_layout()
            plt.suptitle(f'Autoencoder Analysis (Noise std={noise_std})', fontsize=14, y=1.02)
            plt.show()
            fig2, axes2 = plt.subplots(2, num_samples, figsize=(16, 4))
            for i in range(num_samples):
                axes2[0, i].imshow(data[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
                axes2[0, i].set_title('Original Input', fontsize=10)
                axes2[0, i].axis('off')
                axes2[1, i].imshow((data[i] + noise[i]).cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
                axes2[1, i].set_title('Noisy Input', fontsize=10)
                axes2[1, i].axis('off')
            plt.tight_layout()
            plt.suptitle('Input Comparison', fontsize=14, y=1.02)
            plt.show()
            break

def analyze_encoding_results(model, test_loader, noise_levels=[0.0, 0.05, 0.1, 0.2, 0.3]):
    device = get_device()
    model.eval()
    results = {}
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            clean_recon, clean_encoded = model(data)
            clean_error = F.mse_loss(clean_recon, data).item()
            results['clean'] = {
                'reconstruction_error': clean_error,
                'encoding_norm': torch.norm(clean_encoded).item()
            }
            for noise_std in noise_levels:
                if noise_std > 0:
                    noise = torch.randn_like(data) * noise_std
                    noisy_data = data + noise
                    noisy_recon, noisy_encoded = model(noisy_data)
                    noisy_error = F.mse_loss(noisy_recon, data).item()
                    encoding_diff = F.mse_loss(noisy_encoded, clean_encoded).item()
                    results[f'noise_{noise_std}'] = {
                        'reconstruction_error': noisy_error,
                        'encoding_difference': encoding_diff,
                        'encoding_norm': torch.norm(noisy_encoded).item()
                    }
            break
    return results

def plot_analysis(results):
    noise_levels = [float(key.split('_')[1]) for key in results.keys() if key.startswith('noise_')]
    noise_levels.sort()
    reconstruction_errors = [results[f'noise_{std}']['reconstruction_error'] for std in noise_levels]
    encoding_differences = [results[f'noise_{std}']['encoding_difference'] for std in noise_levels]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(noise_levels, reconstruction_errors, 'bo-')
    ax1.axhline(y=results['clean']['reconstruction_error'], color='r', linestyle='--',
                label=f"Clean error: {results['clean']['reconstruction_error']:.4f}")
    ax1.set_xlabel('Noise Standard Deviation')
    ax1.set_ylabel('Reconstruction Error (MSE)')
    ax1.set_title('Reconstruction Error vs Input Noise')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.plot(noise_levels, encoding_differences, 'go-')
    ax2.set_xlabel('Noise Standard Deviation')
    ax2.set_ylabel('Encoding Difference (MSE)')
    ax2.set_title('Encoding Stability vs Input Noise')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show() 