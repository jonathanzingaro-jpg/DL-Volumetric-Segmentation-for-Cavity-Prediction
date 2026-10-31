# Functions/Training/RunEpoch.py

# Import Libraries and Custom Functions
import torch
from Functions.Preprocessing.DataNormalization import *
from Functions.Dataset.CavityDataset import Apply_GPU_Augmentation

# Metric Keys Averaged Per-Sample
_PER_SAMPLE_METRIC_KEYS = ('IoU', 'GrowthIoU', 'FScore', 'Accuracy', 'Tonnage_Error', 'Tonnage_Error_Abs')

def RunEpoch(Model: torch.nn.Module, Optimizer: torch.optim.Optimizer, Dataset: torch.utils.data.DataLoader,
            Loss: torch.nn.Module, Metrics: torch.nn.Module, Device: torch.device, Train_Toggle: bool,
            Binary_Channel_Idx: list = None, Augmentation_Toggle: bool = False,
            Noise_Toggle: bool = False, Noise_Lvl: float = 0.0) -> dict:

    # Set Learning Model
    if Train_Toggle is True:
        Model.train()
    else:
        Model.eval()

    # Declare Variables for Running Sums
    Loss_Cum = 0.0
    Metric_Cum = None
    N_Batches  = 0
    N_Datasets = 0

    # Loop over Batches
    for Input_Batch, Output_Batch, _, Coord_Batch, Backfill_Batch, _, _, Density_Batch, Uranium_Batch in Dataset:

        # Move to Device -- Augmentation/Noise on GPU
        if Train_Toggle is True and (Augmentation_Toggle or Noise_Toggle):
            Input_Batch, Output_Batch, Coord_Batch, Backfill_Batch, Density_Batch, Uranium_Batch = \
                Apply_GPU_Augmentation(
                    Input_Batch, Output_Batch, Coord_Batch, Backfill_Batch, Density_Batch, Uranium_Batch,
                    Device=Device, Binary_Channel_Idx=Binary_Channel_Idx,
                    Augmentation_Toggle=Augmentation_Toggle,
                    Noise_Toggle=Noise_Toggle, Noise_Lvl=Noise_Lvl,
                )

        Input_Batch    = Input_Batch.to(Device, dtype=torch.float32, non_blocking=True)
        Output_Batch   = Output_Batch.to(Device, dtype=torch.float32, non_blocking=True)
        Coord_Batch    = Coord_Batch.to(Device, dtype=torch.float32, non_blocking=True)
        Backfill_Batch = Backfill_Batch.to(Device, dtype=torch.float32, non_blocking=True)
        Density_Batch  = Density_Batch.to(Device, dtype=torch.float32, non_blocking=True)
        Uranium_Batch  = Uranium_Batch.to(Device, dtype=torch.float32, non_blocking=True)

        with torch.set_grad_enabled(Train_Toggle):

            # Forward Pass
            if Train_Toggle is True:
                Optimizer.zero_grad(set_to_none=True)
            Output_Batch_Pred = Model(Input_Batch)      # Output is Logits

            # Compute Loss
            Loss_Batch = Loss(Output_Batch_Pred, Output_Batch, Input_Batch).mean()

            # Backwards Pass
            if Train_Toggle is True:
                Loss_Batch.backward()
                torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=1.0)
                Optimizer.step()

        # Compute Metrics
        Output_Batch_Prob = torch.sigmoid(Output_Batch_Pred)
        Metrics_Batch = Metrics(Output_Batch_Prob, Output_Batch, Density_Batch, Uranium_Batch, Input_Batch)

        # Initialize Cumulative Dict
        if Metric_Cum is None:
            Metric_Cum = {Key: 0.0 for Key in _PER_SAMPLE_METRIC_KEYS if Key in Metrics_Batch}

        # Track Sums
        Loss_Cum += float(Loss_Batch.detach())
        for Key in Metric_Cum:
            Metric_Cum[Key] += float(Metrics_Batch[Key].sum())

        # Track Counts
        N_Batches  += 1
        N_Datasets += Input_Batch.shape[0]

    # Average Metrics
    Results = {'Loss': Loss_Cum / N_Batches}
    for Key, Value in Metric_Cum.items():
        Results[Key] = Value / N_Datasets

    return Results