import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

class DenoisingConvAutoencoder(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), latent_dim=128, noise_factor=0.5):
        super(DenoisingConvAutoencoder, self).__init__()
        self.noise_factor = noise_factor

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.flatten_shape = 256 * (input_shape[1] // 8) * (input_shape[2] // 8)
        self.fc_encode = nn.Linear(self.flatten_shape, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_shape)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, training=True):
        if training:
            noisy_x = x + self.noise_factor * torch.randn_like(x)
            noisy_x = torch.clamp(noisy_x, 0., 1.)
            x = noisy_x

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_encode(x)
        x = self.fc_decode(x)
        x = x.view(x.size(0), 256, 8, 8)
        x = self.decoder(x)
        return x

def train(epocs=50, model, loader, optimizer, criterion):
    epochs = 50
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data, training=True)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                output = model(data, training=False)
                test_loss += criterion(output, data).item()
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}')
    

def run_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    train(epocs=50, model=model, loader=train_loader, optimizer=optimizer, criterion=criterion)