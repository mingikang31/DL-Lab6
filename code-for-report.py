"""Noisy Dataset for MNIST Dataset"""
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset 
from torchvision import transforms, datasets

class NoisyMNIST(Dataset):
    def __init__(self, root='./data', train=True, noise_std=0.3):
        self.noise_std = noise_std
        self.mnist = datasets.MNIST(root=root, train=train, download=True, transform=transforms.ToTensor())

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        noise = torch.randn_like(image) * self.noise_std
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, 0., 1.)
        return noisy_image, image


def measure_psnr(img, img2):
    """Computes the PSNR (Peak Signal-to-Noise Ratio) between two images."""
    mse = nn.MSELoss()(img, img2)
    if mse == 0:
        return float('inf')  # If no noise is present, PSNR is infinite
    max_pixel = 1.0  # Assuming the images are normalized between 0 and 1
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet28(nn.Module):
    def __init__(self):
        super(UNet28, self).__init__()

        # Encoder (Downsampling)
        self.enc1 = self.conv_block(1, 8)
        self.enc2 = self.conv_block(8, 16)
        self.enc3 = self.conv_block(16, 32)
        
        # Bottleneck
        self.bottleneck = self.conv_block(32, 64)

        # Decoder (Upsampling)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # 3x3 -> 6x6
        self.dec3 = self.conv_block(64, 32) # 64 because of concatenation
        
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) # 7x7 -> 14x14
        self.dec2 = self.conv_block(32, 16)
        
        self.up1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2) # 14x14 -> 28x28
        self.dec1 = self.conv_block(16, 8)
        
        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)          # 28x28
        p1 = F.max_pool2d(s1, 2)   # 14x14
        
        s2 = self.enc2(p1)         # 14x14
        p2 = F.max_pool2d(s2, 2)   # 7x7
        
        s3 = self.enc3(p2)         # 7x7
        p3 = F.max_pool2d(s3, 2)   # 3x3
        
        # Bottleneck
        b = self.bottleneck(p3)    # 3x3
        
        # Decoder
        d3 = self.up3(b)
        d3 = F.interpolate(d3, size=s3.shape[2:]) # Force match to 7x7
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)

# Test
model = UNet28()
x = torch.randn(1, 1, 28, 28)
print(model(x).shape)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
#test
model = autoencoder()
x = torch.randn(1, 1, 28, 28)
print(model(x).shape)


"""Main training loop for UNet and Autoencoder"""
import torch 
import torch.nn as nn 
from dataset import NoisyMNIST, measure_psnr
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
import matplotlib.pyplot as plt

# Data Parameters
root = "./Data"
noise_std = 0.3 
batch_size = 32

# Create datasets
train_dataset = NoisyMNIST(root=root, train=True, noise_std=noise_std)
test_dataset = NoisyMNIST(root=root, train=False, noise_std=noise_std)

# create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Initialize Model 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet28().to(device)

# Training Parameters 
num_epochs = 25
optimizer = Adam(model.parameters(), lr=1e-3)
loss_criterion = nn.MSELoss()

# Training and Evaluation Loop 
epoch_times = [] 
train_psnr = [] 
train_loss = []
val_psnr = [] 
val_loss = [] 

for epoch in range(num_epochs):
    start_time = time.time() # save time
    
    # Model training 
    model.train() 
    train_running_loss = 0.0
    train_running_psnr = 0.0

    for noisy_imgs, clean_imgs in train_loader:
        noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = loss_criterion(outputs, clean_imgs)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
        train_running_psnr += measure_psnr(outputs, clean_imgs).item()

    # Validation loop 
    model.eval()
    val_running_loss = 0.0
    val_running_psnr = 0.0

    for noisy_imgs, clean_imgs in test_loader:
        noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

        with torch.no_grad():
            outputs = model(noisy_imgs)
            loss = loss_criterion(outputs, clean_imgs)

        val_running_loss += loss.item()
        val_running_psnr += measure_psnr(outputs, clean_imgs).item()

    # Calculate average metrics for the epoch
    avg_train_loss = train_running_loss / len(train_loader)
    avg_train_psnr = train_running_psnr / len(train_loader)
    avg_val_loss = val_running_loss / len(test_loader)
    avg_val_psnr = val_running_psnr / len(test_loader)

    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)
    train_loss.append(avg_train_loss)
    train_psnr.append(avg_train_psnr)
    val_loss.append(avg_val_loss)
    val_psnr.append(avg_val_psnr)

    # Visualize Validation Images for Noisy, Denoised and Clean Images
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i in range(5):
        axes[0, i].imshow(noisy_imgs[i][0], cmap='gray')
        axes[0, i].set_title('Noisy')
        axes[0, i].axis('off')

        axes[1, i].imshow(outputs[i][0], cmap='gray')
        axes[1, i].set_title('Denoised')
        axes[1, i].axis('off')

        axes[2, i].imshow(clean_imgs[i][0], cmap='gray')
        axes[2, i].set_title('Clean')
        axes[2, i].axis('off')

    plt.suptitle(f'Epoch {epoch+1}')
    plt.savefig(f'./Output/epoch_{epoch+1}.png')
    plt.close()

    # Final Printout for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s - Train Loss: {avg_train_loss:.4f} - Train PSNR: {avg_train_psnr:.2f}dB - Val Loss: {avg_val_loss:.4f} - Val PSNR: {avg_val_psnr:.2f}dB')
    
# Visualize Training and Validation Loss and PSNR Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_psnr, label='Train PSNR')
plt.plot(val_psnr, label='Val PSNR')
plt.title('PSNR Curves')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.savefig('./Output/final_results.png')


import torch 
import torch.nn as nn 
from dataset import NoisyMNIST, measure_psnr
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
import matplotlib.pyplot as plt

# Data Parameters
root = "./Data"
noise_std = 0.3 
batch_size = 32

# Create datasets
train_dataset = NoisyMNIST(root=root, train=True, noise_std=noise_std)
test_dataset = NoisyMNIST(root=root, train=False, noise_std=noise_std)

# create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Initialize Model 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = autoencoder().to(device)

# Training Parameters 
num_epochs = 25
optimizer = Adam(model.parameters(), lr=1e-3)
loss_criterion = nn.MSELoss()

# Training and Evaluation Loop 
epoch_times = [] 
train_psnr = [] 
train_loss = []
val_psnr = [] 
val_loss = [] 

for epoch in range(num_epochs):
    start_time = time.time() # save time
    
    # Model training 
    model.train() 
    train_running_loss = 0.0
    train_running_psnr = 0.0

    for noisy_imgs, clean_imgs in train_loader:
        noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = loss_criterion(outputs, clean_imgs)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
        train_running_psnr += measure_psnr(outputs, clean_imgs).item()

    # Validation loop 
    model.eval()
    val_running_loss = 0.0
    val_running_psnr = 0.0

    for noisy_imgs, clean_imgs in test_loader:
        noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

        with torch.no_grad():
            outputs = model(noisy_imgs)
            loss = loss_criterion(outputs, clean_imgs)

        val_running_loss += loss.item()
        val_running_psnr += measure_psnr(outputs, clean_imgs).item()

    # Calculate average metrics for the epoch
    avg_train_loss = train_running_loss / len(train_loader)
    avg_train_psnr = train_running_psnr / len(train_loader)
    avg_val_loss = val_running_loss / len(test_loader)
    avg_val_psnr = val_running_psnr / len(test_loader)

    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)
    train_loss.append(avg_train_loss)
    train_psnr.append(avg_train_psnr)
    val_loss.append(avg_val_loss)
    val_psnr.append(avg_val_psnr)

    # Visualize Validation Images for Noisy, Denoised and Clean Images
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i in range(5):
        axes[0, i].imshow(noisy_imgs[i][0], cmap='gray')
        axes[0, i].set_title('Noisy')
        axes[0, i].axis('off')

        axes[1, i].imshow(outputs[i][0], cmap='gray')
        axes[1, i].set_title('Denoised')
        axes[1, i].axis('off')

        axes[2, i].imshow(clean_imgs[i][0], cmap='gray')
        axes[2, i].set_title('Clean')
        axes[2, i].axis('off')

    plt.suptitle(f'Epoch {epoch+1}')
    plt.savefig(f'./Output/epoch_{epoch+1}.png')
    plt.close()

    # Final Printout for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s - Train Loss: {avg_train_loss:.4f} - Train PSNR: {avg_train_psnr:.2f}dB - Val Loss: {avg_val_loss:.4f} - Val PSNR: {avg_val_psnr:.2f}dB')
    
# Visualize Training and Validation Loss and PSNR Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_psnr, label='Train PSNR')
plt.plot(val_psnr, label='Val PSNR')
plt.title('PSNR Curves')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.savefig('./Output/final_results.png')

        
