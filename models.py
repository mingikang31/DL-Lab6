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