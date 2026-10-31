#  Functions/Training/Models/UNet3D.py

# Import Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3d(nn.Module):
    """(Conv3d -> BN -> LeakyReLU) x 2 with padding to preserve spatial size."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout3d(p=dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class Down3d(nn.Module):
    """Downscale by 2, then DoubleConv."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DoubleConv3d(in_ch, out_ch, dropout=dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up3d(nn.Module):
    """Upscale by 2, concat skip, then DoubleConv.
    Requires skip channels == in_ch // 2 so that concat(skip, upsampled) == in_ch."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3d(in_ch, out_ch, dropout=dropout)

    @staticmethod
    def _pad_to_match(x, ref):
        diffD = ref.size(2) - x.size(2)
        diffH = ref.size(3) - x.size(3)
        diffW = ref.size(4) - x.size(4)
        return F.pad(x, (diffW // 2, diffW - diffW // 2,
                         diffH // 2, diffH - diffH // 2,
                         diffD // 2, diffD - diffD // 2))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = self._pad_to_match(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv3d(nn.Module):
    """Final 1x1x1 conv to desired out_channels."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric segmentation.

    Incremental Problems (Problem_Type == "Incremental"):
        The model predicts Growth directly -- i.e. the output represents
        logit(P(voxel is newly grown this timestep)), NOT the full absolute
        state. Previous_State ('previous_consumed_cavity') is still fed in
        as an input channel (so the network can see it), but the network is
        never asked to reproduce it; that job happens outside the model
        entirely (see below).

        Growth_Target = Target_State - Previous_State is computed inside
        SegmentationLoss and SegmentationMetrics, from the full Target_State
        each is given -- NOT inside this model. Since Previous_State is
        binary and growth is monotonic, this is an exact {0, 1} mask with
        no ambiguity, and no clamping is required.

        forward() always returns a single plain tensor -- raw logits, with
        NO train/eval branching. For Incremental problems, this tensor
        represents Growth logits specifically, in both training and eval
        mode. Full-state reconstruction (Previous_State + sigmoid(Growth
        logits)) happens entirely in SegmentationMetrics, not here -- this
        keeps the model's output identical regardless of how it's called,
        which is required since RunEpoch calls Model/Loss/Metrics the same
        way for both training and evaluation batches.
    """

    def __init__(self, in_channels, out_channels, Input_Properties, Temp_Toggle, Problem_Type, base_channels=32):
        super().__init__()

        C = base_channels

        # Extract Backfill Index
        self.Backfill_Index = Input_Properties.index('backfill_density')

        # Problem Type
        self.Problem_Type = Problem_Type

        # Previous-State Index -- Only Required for Incremental Problems.
        # 'previous_consumed_cavity' is a binary {0, 1} mask.
        if self.Problem_Type == "Incremental":
            self.Previous_State_Idx = Input_Properties.index('previous_consumed_cavity')
        else:
            self.Previous_State_Idx = None

        # Temperature Scaling Toggle
        self.Temp_Toggle = Temp_Toggle

        # Encoder
        self.inc   = DoubleConv3d(in_channels, C,     dropout=0.0)   # 64^3
        self.down1 = Down3d(C,     C * 2,             dropout=0.0)   # 64^3 -> 32^3
        self.down2 = Down3d(C * 2, C * 4,             dropout=0.0)   # 32^3 -> 16^3
        self.down3 = Down3d(C * 4, C * 8,             dropout=0.0)   # 16^3 -> 8^3

        # Bottleneck
        self.bot   = DoubleConv3d(C * 8, C * 8,       dropout=0.0)   # 8^3

        # Decoder
        self.up2   = Up3d(C * 8,  C * 4,              dropout=0.0)   # 8^3  -> 16^3
        self.up1   = Up3d(C * 4,  C * 2,              dropout=0.0)   # 16^3 -> 32^3
        self.up0   = Up3d(C * 2,  C,                  dropout=0.0)   # 32^3 -> 64^3

        # Output Head: Predicts Growth Logits Directly (Incremental) or
        # Full-State Logits (Non-Incremental).
        self.outc  = OutConv3d(C, out_channels)

        # Boundary Residual Branch
        self.bound_proj = nn.Conv3d(1, out_channels, kernel_size=1, bias=False)

        # Weight Initialisation
        self._init_weights()

        # Temperature Parameter
        self.log_temperature = nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, apply_temperature=None):

        # Permute Channels and Squeeze Temporal Channel
        x = x.squeeze(-2).permute(0, 4, 1, 2, 3)   # (B, Nc, Nx, Ny, Nz)

        # Encoder
        x0 = self.inc(x)        # (N,  C, 64, 64, 64)
        x1 = self.down1(x0)     # (N, 2C, 32, 32, 32)
        x2 = self.down2(x1)     # (N, 4C, 16, 16, 16)
        x3 = self.down3(x2)     # (N, 8C,  8,  8,  8)

        # Bottleneck
        xb = self.bot(x3)       # (N, 8C,  8,  8,  8)

        # Decoder
        y2 = self.up2(xb, x2)   # (N, 4C, 16, 16, 16)
        y1 = self.up1(y2, x1)   # (N, 2C, 32, 32, 32)
        y0 = self.up0(y1, x0)   # (N,  C, 64, 64, 64)

        # Output Logits -- Growth Logits (Incremental) or Full-State Logits
        logits = self.outc(y0)

        # Backfill Residual
        boundary = x[:, self.Backfill_Index, ...].unsqueeze(1)
        logits   = logits + self.bound_proj(boundary)

        # Temperature Scaling
        use_T = self.Temp_Toggle if apply_temperature is None else apply_temperature
        if use_T:
            logits = logits / self.temperature

        return logits.permute(0, 2, 3, 4, 1).unsqueeze(-2)   # (B, Nx, Ny, Nz, 1, Nc)