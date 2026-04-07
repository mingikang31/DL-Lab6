
import torch 
import torch.nn as nn 
from dataset import NoisyMNIST, measure_psnr
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
import matplotlib.pyplot as plt
from models import unet, autoencoder

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
model = unet().to(device)

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

        