# Functions/Data/CavityDataset

# Import Libraries
import os
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import torch.nn.functional as F
from pathlib import Path
from typing import List


class DiskBatchDataset(IterableDataset):
    """
    Streams pre-normalized, pre-batched .pt files (written by DataNormalization.Process)
    straight from disk. 
    """
    def __init__(self, Split_Dir: str, Internal_Batch_Size: int,
                 Shuffle_Within_File: bool = True):
        self.Files = sorted(Path(Split_Dir).glob("batch_*.pt"))
        if not self.Files:
            raise FileNotFoundError(f"No batch files found in {Split_Dir}")
        self.Internal_Batch_Size = Internal_Batch_Size
        self.Shuffle_Within_File = Shuffle_Within_File

    def __len__(self):
        """Approximate number of sub-batches per epoch."""
        return sum(
            -(-torch.load(f, map_location='cpu')['Input'].size(0) // self.Internal_Batch_Size)
            for f in self.Files
        )

    def __iter__(self):
        Info = get_worker_info()
        Files = self.Files if Info is None else self.Files[Info.id::Info.num_workers]

        for File in Files:

            Batch = torch.load(File, map_location='cpu')
            N = Batch['Input'].size(0)
            Order = torch.randperm(N) if self.Shuffle_Within_File else torch.arange(N)

            for Start in range(0, N, self.Internal_Batch_Size):
                Idx = Order[Start:Start + self.Internal_Batch_Size]

                Input    = Batch['Input'][Idx]
                Output   = Batch['Output'][Idx]
                Coord    = Batch['Coord'][Idx]
                Backfill = Batch['Backfill'][Idx]
                Density  = Batch['Density'][Idx]
                Uranium  = Batch['Uranium'][Idx]
                Name         = [Batch['Cavity_Name'][i] for i in Idx]
                Dataset_Type = [Batch['Dataset_Type'][i] for i in Idx]
                Field_Mask   = torch.tensor([t == 'Field' for t in Dataset_Type], dtype=torch.float32)

                yield (Input, Output, Name, Coord, Backfill, Dataset_Type, Field_Mask, Density, Uranium)


def Add_Noise(Input: torch.Tensor, Noise_Std: float,
              Binary_Channel_Idx: List[int] = None, Probability: float = 0.5):
    """
    Applies multiplicative Gaussian noise to Input: Input * (1 + N(0, Noise_Std)).

    Applied per-sample with the given probability. Kept separate from Augmentation so
    noise and geometric augmentation can be toggled independently in the config file.
    """
    if Noise_Std is None or Noise_Std <= 0:
        return Input

    Device      = Input.device
    Batch_Size  = Input.shape[0]
    Apply_Noise = torch.rand(Batch_Size, device=Device) > (1.0 - Probability)

    Noise = 1.0 + torch.randn(Input.shape, device=Device) * Noise_Std
    if Binary_Channel_Idx:
        Noise[..., Binary_Channel_Idx] = 1.0   # No Multiplier

    Noise_Mask = Apply_Noise.view(Batch_Size, *([1] * (Input.dim() - 1)))

    return torch.where(Noise_Mask, Input * Noise, Input)


def Augmentation(Input: torch.Tensor, Output: torch.Tensor, Coord: torch.Tensor,
                 Backfill: torch.Tensor, Density: torch.Tensor, Uranium: torch.Tensor,
                 Binary_Channel_Idx: List[int] = None,
                 Max_Shift_Voxels_XY=5, Max_Shift_Voxels_Z=3):
    """
    Applies randomized, per-sample GEOMETRIC augmentation to a batch of 3D volumetric data.

    For each sample in the batch, independently and with some probability:
      - Applies a random rotation (about the Z axis) and XYZ translation via an
        affine grid transform, identically across Input, Output, Coord, Backfill,
        Density, and Uranium, and identically across all Nt time steps and channels.
      - Applies random flips along the X and/or Y spatial axes, identically across
        all six tensors.

    """
    Device = Input.device
    B, D, H, W, Nt, Nc = Input.shape
    _, _, _, _, Nt_Out, Nc_Out = Output.shape
    _, _, _, _, Nt_Cd, Nc_Cd   = Coord.shape
    _, _, _, _, Nt_Bf          = Backfill.shape
    _, _, _, _, Nt_Df          = Density.shape
    _, _, _, _, Nt_Ur          = Uranium.shape

    Binary_Channel_Idx = Binary_Channel_Idx or []
    Continuous_Idx = [c for c in range(Nc) if c not in Binary_Channel_Idx]

    # Split Input
    if Binary_Channel_Idx:
        Input_Cont = Input[..., Continuous_Idx]         # (B,D,H,W,Nt,len(cont))
        Input_Bin  = Input[..., Binary_Channel_Idx]     # (B,D,H,W,Nt,len(bin))
        Input_Cont_C = Input_Cont.permute(0, 4, 5, 1, 2, 3).reshape(B, Nt * len(Continuous_Idx), D, H, W)
        Input_Bin_C  = Input_Bin.permute(0, 4, 5, 1, 2, 3).reshape(B, Nt * len(Binary_Channel_Idx), D, H, W)
        Ref_Shape = (B, Nt * Nc, D, H, W)   # shape reference only, for affine_grid
    else:
        Input_C = Input.permute(0, 4, 5, 1, 2, 3).reshape(B, Nt * Nc, D, H, W)
        Ref_Shape = Input_C.shape

    Output_C   = Output.permute(0, 4, 5, 1, 2, 3).reshape(B, Nt_Out * Nc_Out, D, H, W)
    Coord_C    = Coord.permute(0, 4, 5, 1, 2, 3).reshape(B, Nt_Cd * Nc_Cd, D, H, W)
    Backfill_C = Backfill.permute(0, 4, 1, 2, 3).reshape(B, Nt_Bf, D, H, W)
    Density_C  = Density.permute(0, 4, 1, 2, 3).reshape(B, Nt_Df, D, H, W)
    Uranium_C  = Uranium.permute(0, 4, 1, 2, 3).reshape(B, Nt_Ur, D, H, W)

    # Per-Sample Augmentation Masks
    Apply_Affine = torch.rand(B, device=Device) > 0.4
    Flip_X       = torch.rand(B, device=Device) > 0.5
    Flip_Y       = torch.rand(B, device=Device) > 0.5

    # Per-sample random angles and translations
    angles = torch.rand(B, device=Device) * 2 * torch.pi * Apply_Affine.float()
    tx     = (torch.rand(B, device=Device) * 2 - 1) * (Max_Shift_Voxels_XY / (W / 2)) * Apply_Affine.float()
    ty     = (torch.rand(B, device=Device) * 2 - 1) * (Max_Shift_Voxels_XY / (H / 2)) * Apply_Affine.float()
    tz     = (torch.rand(B, device=Device) * 2 - 1) * (Max_Shift_Voxels_Z  / (D / 2)) * Apply_Affine.float()

    # Build Per-Sample Affine Matrices
    theta = torch.zeros(B, 3, 4, device=Device)
    theta[:, 0, 0] =  torch.cos(angles)
    theta[:, 0, 1] = -torch.sin(angles)
    theta[:, 1, 0] =  torch.sin(angles)
    theta[:, 1, 1] =  torch.cos(angles)
    theta[:, 2, 2] =  1.0
    theta[:, 0, 3] =  tx
    theta[:, 1, 3] =  ty
    theta[:, 2, 3] =  tz

    grid = F.affine_grid(theta, Ref_Shape, align_corners=False)

    if Binary_Channel_Idx:
        Input_Cont_C = F.grid_sample(Input_Cont_C, grid, mode='bilinear', padding_mode='border', align_corners=False)
        Input_Bin_C  = F.grid_sample(Input_Bin_C,  grid, mode='nearest',  padding_mode='border', align_corners=False)
    else:
        Input_C = F.grid_sample(Input_C, grid, mode='bilinear', padding_mode='border', align_corners=False)

    Output_C   = F.grid_sample(Output_C,   grid, mode='nearest',  padding_mode='border', align_corners=False)
    Coord_C    = F.grid_sample(Coord_C,    grid, mode='nearest',  padding_mode='border', align_corners=False)
    Backfill_C = F.grid_sample(Backfill_C, grid, mode='nearest',  padding_mode='border', align_corners=False)
    Density_C  = F.grid_sample(Density_C,  grid, mode='bilinear', padding_mode='border', align_corners=False)
    Uranium_C  = F.grid_sample(Uranium_C,  grid, mode='bilinear', padding_mode='border', align_corners=False)

    # Reassemble Input_C
    if Binary_Channel_Idx:
        Input_Cont_Un = Input_Cont_C.reshape(B, Nt, len(Continuous_Idx), D, H, W)
        Input_Bin_Un  = Input_Bin_C.reshape(B, Nt, len(Binary_Channel_Idx), D, H, W)
        Input_Un = torch.empty(B, Nt, Nc, D, H, W, device=Device, dtype=Input.dtype)
        Input_Un[:, :, Continuous_Idx]     = Input_Cont_Un
        Input_Un[:, :, Binary_Channel_Idx] = Input_Bin_Un
        Input_C = Input_Un.reshape(B, Nt * Nc, D, H, W)

    # Vectorized Per-Sample XY Flips
    Flip_X_Mask = Flip_X.view(B, 1, 1, 1, 1)
    Flip_Y_Mask = Flip_Y.view(B, 1, 1, 1, 1)

    Input_C    = torch.where(Flip_X_Mask, Input_C.flip(dims=[4]),    Input_C)
    Output_C   = torch.where(Flip_X_Mask, Output_C.flip(dims=[4]),   Output_C)
    Coord_C    = torch.where(Flip_X_Mask, Coord_C.flip(dims=[4]),    Coord_C)
    Backfill_C = torch.where(Flip_X_Mask, Backfill_C.flip(dims=[4]), Backfill_C)
    Density_C  = torch.where(Flip_X_Mask, Density_C.flip(dims=[4]),  Density_C)
    Uranium_C  = torch.where(Flip_X_Mask, Uranium_C.flip(dims=[4]),  Uranium_C)

    Input_C    = torch.where(Flip_Y_Mask, Input_C.flip(dims=[3]),    Input_C)
    Output_C   = torch.where(Flip_Y_Mask, Output_C.flip(dims=[3]),   Output_C)
    Coord_C    = torch.where(Flip_Y_Mask, Coord_C.flip(dims=[3]),    Coord_C)
    Backfill_C = torch.where(Flip_Y_Mask, Backfill_C.flip(dims=[3]), Backfill_C)
    Density_C  = torch.where(Flip_Y_Mask, Density_C.flip(dims=[3]),  Density_C)
    Uranium_C  = torch.where(Flip_Y_Mask, Uranium_C.flip(dims=[3]),  Uranium_C)

    # Unfold Channels
    Input    = Input_C.reshape(B, Nt, Nc, D, H, W).permute(0, 3, 4, 5, 1, 2)
    Output   = Output_C.reshape(B, Nt_Out, Nc_Out, D, H, W).permute(0, 3, 4, 5, 1, 2)
    Coord    = Coord_C.reshape(B, Nt_Cd, Nc_Cd, D, H, W).permute(0, 3, 4, 5, 1, 2)
    Backfill = Backfill_C.reshape(B, Nt_Bf, D, H, W).permute(0, 2, 3, 4, 1)
    Density  = Density_C.reshape(B, Nt_Df, D, H, W).permute(0, 2, 3, 4, 1)
    Uranium  = Uranium_C.reshape(B, Nt_Ur, D, H, W).permute(0, 2, 3, 4, 1)

    return Input, Output, Coord, Backfill, Density, Uranium


def Apply_GPU_Augmentation(Input, Output, Coord, Backfill, Density, Uranium,
                           Device, Binary_Channel_Idx: List[int] = None,
                           Augmentation_Toggle: bool = False,
                           Noise_Toggle: bool = False, Noise_Lvl: float = 0.0):
    """
    Moves a batch to Device, then applies Augmentation and Add_Noise.
    """
    Input    = Input.to(Device, dtype=torch.float32, non_blocking=True)
    Output   = Output.to(Device, dtype=torch.float32, non_blocking=True)
    Coord    = Coord.to(Device, dtype=torch.float32, non_blocking=True)
    Backfill = Backfill.to(Device, dtype=torch.float32, non_blocking=True)
    Density  = Density.to(Device, dtype=torch.float32, non_blocking=True)
    Uranium  = Uranium.to(Device, dtype=torch.float32, non_blocking=True)

    if Augmentation_Toggle:
        Input, Output, Coord, Backfill, Density, Uranium = Augmentation(
            Input, Output, Coord, Backfill, Density, Uranium,
            Binary_Channel_Idx=Binary_Channel_Idx)

    if Noise_Toggle:
        Input = Add_Noise(Input, Noise_Lvl, Binary_Channel_Idx=Binary_Channel_Idx)

    return Input, Output, Coord, Backfill, Density, Uranium


## Build Dataloaders
def BuildLoaders(ProcessedDir: str, Data_Config: dict, Input_Properties: List[str], Num_Workers: int = 4):
    """
    Construct disk-streaming dataloaders directly from Processed Batches
    (ProcessedDir/{Train,Validation,Test}/batch_*.pt), as written by
    DataNormalization.Process. 
    """
    Binary_Channel_Names = ['previous_consumed_cavity', 'backfill_density',
                            'pilot_hole_distance', 'nozzle_angle', 'jetting_time',
                            'jetting_pumps', 'rotation_speed']
    Binary_Channel_Idx = [Input_Properties.index(n) for n in Binary_Channel_Names
                          if n in Input_Properties]

    Train_Set = DiskBatchDataset(
        os.path.join(ProcessedDir, 'Train'),
        Internal_Batch_Size=Data_Config['Batch_Size'],
        Shuffle_Within_File=True,
    )
    Train_Eval_Set = DiskBatchDataset(
        os.path.join(ProcessedDir, 'Train'),
        Internal_Batch_Size=Data_Config['Batch_Size'],
        Shuffle_Within_File=False,
    )
    Valid_Set = DiskBatchDataset(
        os.path.join(ProcessedDir, 'Validation'),
        Internal_Batch_Size=Data_Config['Batch_Size'],
        Shuffle_Within_File=False,
    )
    Test_Set = DiskBatchDataset(
        os.path.join(ProcessedDir, 'Test'),
        Internal_Batch_Size=Data_Config['Batch_Size'],
        Shuffle_Within_File=False,
    )

    Loader_Args = dict(batch_size=None, num_workers=Num_Workers,
                       pin_memory=True,
                       prefetch_factor=2 if Num_Workers > 0 else None,
                       persistent_workers=Num_Workers > 0)

    return {
        'Train':      DataLoader(Train_Set,      **Loader_Args),
        'Train_Eval': DataLoader(Train_Eval_Set, **Loader_Args),
        'Valid':      DataLoader(Valid_Set,      **Loader_Args),
        'Test':       DataLoader(Test_Set,       **Loader_Args),
        'Binary_Channel_Idx': Binary_Channel_Idx,
    }