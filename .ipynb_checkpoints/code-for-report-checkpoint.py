"""Noisy Dataset for MNIST Dataset"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class NoisyMNIST(Dataset):
    def __init__(self, root="./Data", train=True, noise_std=0.3):
        self.noise_std = noise_std
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, _ = self.mnist[idx]
        noise = torch.randn_like(image) * self.noise_std
        noisy_image = torch.clamp(image + noise, 0.0, 1.0)
        return noisy_image, image

def measure_psnr(img, img2):
    mse = torch.mean((img - img2) ** 2)
    if mse.item() == 0:
        return float("inf")
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet28(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = self.conv_block(1, 8)
        self.enc2 = self.conv_block(8, 16)
        self.enc3 = self.conv_block(16, 32)

        self.bottleneck = self.conv_block(32, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(64, 32)

        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(32, 16)

        self.up1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(16, 8)

        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        s1 = self.enc1(x)
        p1 = F.max_pool2d(s1, 2)

        s2 = self.enc2(p1)
        p2 = F.max_pool2d(s2, 2)

        s3 = self.enc3(p2)
        p3 = F.max_pool2d(s3, 2)

        b = self.bottleneck(p3)

        d3 = self.up3(b)
        d3 = F.interpolate(d3, size=s3.shape[2:], mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

"""Denoising Training"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("./Output_Den", exist_ok=True)

mu_values = [0.1, 0.3, 0.6]
conditions = ["Mild", "Medium", "Severe"]
epochs = 20
batch_size = 32

for mu, condition in zip(mu_values, conditions):
    print(f"\n{'='*50}")
    print(f"STARTING NOISE LEVEL: {condition} (mu={mu})")
    print(f"{'='*50}")

    train_dataset = NoisyMNIST(root="./Data", train=True, noise_std=mu)
    test_dataset = NoisyMNIST(root="./Data", train=False, noise_std=mu)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"--- Training U-Net ({condition}) ---")
    unet_model = UNet28().to(device)
    unet_optimizer = Adam(unet_model.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    for epoch in range(epochs):
        unet_model.train()
        train_running_loss = 0.0
        train_running_psnr = 0.0

        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            unet_optimizer.zero_grad()
            outputs = unet_model(noisy_imgs)
            loss = loss_criterion(outputs, clean_imgs)
            loss.backward()
            unet_optimizer.step()

            train_running_loss += loss.item()
            train_running_psnr += measure_psnr(outputs.detach(), clean_imgs.detach()).item()

        unet_model.eval()
        val_running_loss = 0.0
        val_running_psnr = 0.0

        with torch.no_grad():
            for noisy_imgs, clean_imgs in test_loader:
                noisy_imgs = noisy_imgs.to(device)
                clean_imgs = clean_imgs.to(device)
                outputs = unet_model(noisy_imgs)
                loss = loss_criterion(outputs, clean_imgs)

                val_running_loss += loss.item()
                val_running_psnr += measure_psnr(outputs, clean_imgs).item()

        avg_train_loss = train_running_loss / len(train_loader)
        avg_train_psnr = train_running_psnr / len(train_loader)
        avg_val_loss = val_running_loss / len(test_loader)
        avg_val_psnr = val_running_psnr / len(test_loader)

        print(
            f"U-Net Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {avg_train_loss:.8f} - Train PSNR: {avg_train_psnr:.2f}dB - "
            f"Val Loss: {avg_val_loss:.8f} - Val PSNR: {avg_val_psnr:.2f}dB"
        )

        num_show = min(5, noisy_imgs.size(0))
        fig, axes = plt.subplots(3, num_show, figsize=(3 * num_show, 9))
        for i in range(num_show):
            axes[0, i].imshow(noisy_imgs[i][0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[0, i].set_title("Noisy")
            axes[0, i].axis("off")

            axes[1, i].imshow(outputs[i][0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[1, i].set_title("Denoised")
            axes[1, i].axis("off")

            axes[2, i].imshow(clean_imgs[i][0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[2, i].set_title("Clean")
            axes[2, i].axis("off")

        plt.suptitle(f"U-Net {condition} - Epoch {epoch+1}")
        plt.tight_layout()
        plt.savefig(f"./Output_Den/UNet_{condition}_epoch_{epoch+1}.png")
        plt.close()

    print(f"\n--- Training Autoencoder ({condition}) ---")
    ae_model = Autoencoder().to(device)
    ae_optimizer = Adam(ae_model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        ae_model.train()
        train_running_loss = 0.0
        train_running_psnr = 0.0

        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            ae_optimizer.zero_grad()
            outputs = ae_model(noisy_imgs)
            loss = loss_criterion(outputs, clean_imgs)
            loss.backward()
            ae_optimizer.step()

            train_running_loss += loss.item()
            train_running_psnr += measure_psnr(outputs.detach(), clean_imgs.detach()).item()

        ae_model.eval()
        val_running_loss = 0.0
        val_running_psnr = 0.0

        with torch.no_grad():
            for noisy_imgs, clean_imgs in test_loader:
                noisy_imgs = noisy_imgs.to(device)
                clean_imgs = clean_imgs.to(device)
                outputs = ae_model(noisy_imgs)
                loss = loss_criterion(outputs, clean_imgs)

                val_running_loss += loss.item()
                val_running_psnr += measure_psnr(outputs, clean_imgs).item()

        avg_train_loss = train_running_loss / len(train_loader)
        avg_train_psnr = train_running_psnr / len(train_loader)
        avg_val_loss = val_running_loss / len(test_loader)
        avg_val_psnr = val_running_psnr / len(test_loader)

        print(
            f"Autoencoder Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {avg_train_loss:.8f} - Train PSNR: {avg_train_psnr:.2f}dB - "
            f"Val Loss: {avg_val_loss:.8f} - Val PSNR: {avg_val_psnr:.2f}dB"
        )

        num_show = min(5, noisy_imgs.size(0))
        fig, axes = plt.subplots(3, num_show, figsize=(3 * num_show, 9))
        for i in range(num_show):
            axes[0, i].imshow(noisy_imgs[i][0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[0, i].set_title("Noisy")
            axes[0, i].axis("off")

            axes[1, i].imshow(outputs[i][0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[1, i].set_title("Denoised")
            axes[1, i].axis("off")

            axes[2, i].imshow(clean_imgs[i][0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[2, i].set_title("Clean")
            axes[2, i].axis("off")

        plt.suptitle(f"Autoencoder {condition} - Epoch {epoch+1}")
        plt.tight_layout()
        plt.savefig(f"./Output_Den/Autoencoder_{condition}_epoch_{epoch+1}.png")
        plt.close()


"""Horse Segmentation Training"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F

class HorseDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(224, 224)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        image_map = {os.path.splitext(f)[0]: f for f in self.images}
        mask_map = {os.path.splitext(f)[0]: f for f in self.masks}
        common_keys = sorted(set(image_map) & set(mask_map))
        self.pairs = [(image_map[k], mask_map[k]) for k in common_keys]

        if len(self.pairs) == 0:
            raise ValueError("No matched image/mask filename pairs found.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_name, mask_name = self.pairs[idx]

        img_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, self.img_size)
        mask = TF.resize(mask, self.img_size, interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        mask = (mask > 0).long()

        return image, mask

class UNetSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(3, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)

        self.bottleneck = self.conv_block(64, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(32, 16)

        self.final = nn.Conv2d(16, 2, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        s1 = self.enc1(x)
        p1 = F.max_pool2d(s1, 2)

        s2 = self.enc2(p1)
        p2 = F.max_pool2d(s2, 2)

        s3 = self.enc3(p2)
        p3 = F.max_pool2d(s3, 2)

        b = self.bottleneck(p3)

        d3 = self.up3(b)
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)

class AutoencoderSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
epochs = 20

os.makedirs("./Output_Seg", exist_ok=True)

image_directory = "./Data/horses/horse"
mask_directory = "./Data/horses/mask"

full_dataset = HorseDataset(image_dir=image_directory, mask_dir=mask_directory, img_size=(224, 224))
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

unet_model = UNetSeg().to(device)
ae_model = AutoencoderSeg().to(device)

loss_criterion = nn.CrossEntropyLoss()
unet_optimizer = Adam(unet_model.parameters(), lr=1e-3)
ae_optimizer = Adam(ae_model.parameters(), lr=1e-3)

def calculate_iou(logits, labels, num_classes=2, eps=1e-6):
    preds = torch.argmax(logits, dim=1)
    ious = []

    for c in range(num_classes):
        pred_c = preds == c
        label_c = labels == c

        intersection = (pred_c & label_c).sum(dim=(1, 2)).float()
        union = (pred_c | label_c).sum(dim=(1, 2)).float()
        iou = (intersection + eps) / (union + eps)
        ious.append(iou)

    mean_iou = torch.stack(ious, dim=0).mean(dim=0)
    return mean_iou.mean().item()

def train_model(model, optimizer, model_name):
    print(f"\n{'='*50}\n--- Training {model_name} ---\n{'='*50}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_iou = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou += calculate_iou(outputs, masks)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = loss_criterion(outputs, masks)

                val_loss += loss.item()
                val_iou += calculate_iou(outputs, masks)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        avg_val_iou = val_iou / len(test_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.8f}, Train IoU: {avg_train_iou:.8f} | "
            f"Val Loss: {avg_val_loss:.8f}, Val IoU: {avg_val_iou:.8f}"
        )

        predicted_masks = torch.argmax(outputs, dim=1)

        num_show = min(3, images.size(0))
        fig, axes = plt.subplots(3, num_show, figsize=(4 * num_show, 9))
        for i in range(num_show):
            img_to_plot = images[i].cpu().permute(1, 2, 0).numpy()
            axes[0, i].imshow(img_to_plot)
            axes[0, i].set_title("Original Image")
            axes[0, i].axis("off")

            axes[1, i].imshow(predicted_masks[i].cpu().numpy(), cmap="gray")
            axes[1, i].set_title("Predicted Mask")
            axes[1, i].axis("off")

            axes[2, i].imshow(masks[i].cpu().numpy(), cmap="gray")
            axes[2, i].set_title("Ground Truth")
            axes[2, i].axis("off")

        plt.suptitle(f"{model_name} Segmentation - Epoch {epoch+1}")
        plt.tight_layout()
        plt.savefig(f"./Output_Seg/{model_name}_epoch_{epoch+1}.png")
        plt.close()

train_model(unet_model, unet_optimizer, "UNet")
train_model(ae_model, ae_optimizer, "Autoencoder")