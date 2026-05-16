import torch
import torch.nn as nn
import torch.nn.functional as F


class MedicalDetectorCNN(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(MedicalDetectorCNN, self).__init__()

        # Encoder (Downsampling)
        self.inc = self.double_conv(n_channels, 64)
        self.down1 = self.down(64, 128)
        self.down2 = self.down(128, 256)

        # Decoder (Upsampling)
        self.up1 = self.up(256, 128)
        self.up2 = self.up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def down(self, in_c, out_c):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_c, out_c)
        )

    def up(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3)
        x = self.up2(x)
        logits = self.outc(x)
        return self.sigmoid(logits)