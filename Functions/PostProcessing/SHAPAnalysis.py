# Functions/PostProcessing/SHAPAnalysis.py

# Import Libraries
import os
import numpy
from scipy.io import savemat

# Import Custom Functions
from Functions.Imports.Training import *


def ShapAnalysis(Dataset: torch.utils.data.DataLoader, Model: torch.nn.Module, Result_Path: str,
                 Input_Keys: list, Dataset_Type: str, Save_Maps: bool = True, Contour_Level: float = 0.5,
                 IG_Batch_Size: int = 16):
    """Integrated Gradients over Dataset.

    Prints the mean |IG| per input channel, and with Save_Maps writes one
    IG_<name>.mat per sample and time step.
    """
    # Declare Device
    Device = next(Model.parameters()).device
    if isinstance(Model, torch.nn.DataParallel):
        Model = Model.module
    Model.eval()

    # Forward Wrapper
    def forward_all_outputs(x):
        """Scalar target for Integrated Gradients: total predicted occupancy
        (probability mass, bounded in [0, 1] per voxel)."""
        out = torch.sigmoid(Model(x))
        return out.reshape(out.size(0), -1).sum(dim=1)

    IG_Method = captum.attr.IntegratedGradients(forward_all_outputs)

    Map_Dir = os.path.join(Result_Path, 'IG')
    if Save_Maps:
        os.makedirs(Map_Dir, exist_ok=True)

    nb = 0
    N_Saved = 0
    IG_Store = torch.zeros(1, len(Input_Keys), device=Device)
    Baselines = None

    with torch.enable_grad():
        for Input_Batch, Output_Batch, Dataset_Name_Batch, Coord_Batch, Backfill_Batch, _, _, _, _ in Dataset:
            Input_Batch = Input_Batch.to(Device, dtype=torch.float32)

            Full_Batch_Size = Input_Batch.shape[0]
            Batch_Size = Full_Batch_Size if IG_Batch_Size is None else min(IG_Batch_Size, Full_Batch_Size)

            IG_Chunks = []

            for Start in range(0, Full_Batch_Size, Batch_Size):
                Input_Sub = Input_Batch[Start:Start + Batch_Size]

                if Baselines is None or Baselines.shape != Input_Sub.shape:
                    Baselines = torch.zeros_like(Input_Sub)

                IG_Sub, _ = IG_Method.attribute(
                    inputs=Input_Sub,
                    baselines=Baselines,
                    method='gausslegendre',
                    target=None,
                    n_steps=25,
                    return_convergence_delta=True,
                    internal_batch_size=Input_Sub.shape[0]
                )
                IG_Chunks.append(IG_Sub)

            IG_Batch = torch.cat(IG_Chunks, dim=0)

            nb += 1
            Batch_Mean = torch.mean(torch.abs(IG_Batch), dim=(0, 1, 2, 3, 4), keepdim=False)
            IG_Store += (Batch_Mean - IG_Store) / nb

            # Save Per-Cavity Maps
            if Save_Maps:
                with torch.no_grad():
                    Pred_Batch = torch.sigmoid(Model(Input_Batch))

                N_Saved += _Save_IG_Maps(IG_Batch.detach(), Pred_Batch.detach(),
                                         Output_Batch, Backfill_Batch, Dataset_Name_Batch,
                                         Input_Keys, Map_Dir, Contour_Level)

    # Print Results as a Table
    print(f'\nAverage Integrated Gradients for {Dataset_Type}:\n')

    IG_Values = IG_Store.squeeze(0).detach().cpu().tolist()

    # Sort by Importance (Descending)
    Sorted_Pairs = sorted(zip(Input_Keys, IG_Values), key=lambda pair: pair[1], reverse=True)

    print(f"{'Property':<30}{'Avg |IG|':>15}")
    print("-" * 45)
    for Name, Value in Sorted_Pairs:
        print(f"{Name:<30}{Value:>15.6f}")

    if Save_Maps:
        print(f"\nWrote {N_Saved} IG map file(s) to {Map_Dir}")

    return


def _Save_IG_Maps(IG_Batch, Pred_Batch, Output_Batch, Backfill_Batch, Dataset_Name_Batch,
                  Input_Keys: list, Map_Dir: str, Contour_Level: float) -> int:
    """One IG_<dataset>_T<t>.mat per sample and time step."""
    # (B, Nx, Ny, Nz, Nt, C) in, channel last
    IG_Np    = IG_Batch.cpu().numpy()
    Pred_Np  = Pred_Batch.cpu().numpy()
    Truth_Np = Output_Batch.cpu().numpy()
    Back_Np  = Backfill_Batch.cpu().numpy()

    Nb, Nt = IG_Np.shape[0], IG_Np.shape[4]
    Written = 0

    for i in range(Nb):
        for t in range(Nt):

            Name = f"{Dataset_Name_Batch[i]}_T{t}"

            # (Nx, Ny, Nz, C) -> (C, Nx, Ny, Nz)
            Signed_Map = numpy.transpose(IG_Np[i, :, :, :, t, :], (3, 0, 1, 2))

            Data_Structure = {
                'name':             str(Name),
                'keys':             numpy.array(list(Input_Keys), dtype=object),
                'signed_map':       Signed_Map.astype(numpy.float32),
                'pred_occupancy':   Pred_Np[i, :, :, :, t, 0].astype(numpy.float32),
                'truth_occupancy':  Truth_Np[i, :, :, :, t, 0].astype(numpy.float32),
                'backfill_mask':    Back_Np[i, :, :, :, t].astype(numpy.float32),
                'meta': {'contour_level': float(Contour_Level),
                         'n_channels':    int(len(Input_Keys)),
                         'shape':         numpy.array(Signed_Map.shape[1:], dtype=numpy.float64)},
            }

            savemat(os.path.join(Map_Dir, f'IG_{Name}.mat'), {'IG': Data_Structure},
                    do_compression=True)
            Written += 1

    return Written