# Functions/Training/Models/3DCNN.py

# Import Libraries
from Functions.Imports.Training import *

class CustomCNN(nn.Module):
    """
    A simple 3D CNN that keeps the same spatial size (64³) using padding.
    Input:  (B, In_Channels, 64, 64, 64)
    Output: (B, Out_Channels, 64, 64, 64)
    """
    def __init__(self, in_channels, out_channels):
        super(CustomCNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)