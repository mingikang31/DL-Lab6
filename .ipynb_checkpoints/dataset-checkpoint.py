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


