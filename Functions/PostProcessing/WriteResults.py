## Functions/PostProcessing/WriteResults.py

## Import Libraries
import os
# import vtk
import torch
import pandas as pd
# from vtkmodules.util.numpy_support import numpy_to_vtk
from Functions.Preprocessing.DataNormalization import *


# def _Add_Scalar_Array(Image_Data: "vtk.vtkImageData", Volume, Name: str) -> None:
#     """
#     Summary: Flattens a (nx, ny, nz) numpy volume in Fortran order and attaches
#     it to Image_Data as a named single-component VTK float array, using the
#     numpy-to-VTK bridge instead of a per-voxel Python loop.
#     """
#     Flat = Volume.ravel(order="F")
#     Arr = numpy_to_vtk(Flat, deep=True, array_type=vtk.VTK_FLOAT)
#     Arr.SetName(Name)
#     Image_Data.GetPointData().AddArray(Arr)


# ## Custom Functions
# def WriteToVTK(IG_Batch: torch.Tensor, Dataset_Name_Batch: list, Input_Properties: dict, Result_Path: str,
#                Output_Batch: torch.Tensor, Output_Batch_Pred: torch.Tensor, Input_Batch: torch.Tensor) -> None:
#     """
#     Summary: Writes one .vti (VTK ImageData) file per sample in the batch,
#     containing integrated-gradient volumes, input property volumes, ground
#     truth, and predicted cavern volumes as named point-data scalar arrays.
#     """
#     # Move Tensors to Numpy
#     IG_Batch_Np          = IG_Batch.detach().cpu().numpy()          # (B, inC, nx, ny, nz)
#     Input_Batch_Np       = Input_Batch.detach().cpu().numpy()       # (B, inC, nx, ny, nz)
#     Output_Batch_Np      = Output_Batch.detach().cpu().numpy()      # (B, outC, nx, ny, nz)
#     Output_Batch_Pred_Np = Output_Batch_Pred.detach().cpu().numpy() # (B, outC, nx, ny, nz)
#     B, InC, Nx, Ny, Nz = IG_Batch_Np.shape

#     # Define Voxel Spacing and Origin
#     Dx = Dy = Dz = 0.25
#     Ox = Oy = Oz = 0.0

#     Vtk_Dir = os.path.join(Result_Path, "VTK")
#     os.makedirs(Vtk_Dir, exist_ok=True)

#     for Bi in range(B):
#         Sample_Name = str(Dataset_Name_Batch[Bi])

#         # Set Origin of Image
#         Image_Data = vtk.vtkImageData()
#         Image_Data.SetDimensions(Nx, Ny, Nz)
#         Image_Data.SetSpacing(Dx, Dy, Dz)
#         Image_Data.SetOrigin(Ox, Oy, Oz)

#         # Write Integrated Gradients
#         for C in range(InC):
#             Prop_Name = Input_Properties["Names"][C]
#             _Add_Scalar_Array(Image_Data, IG_Batch_Np[Bi, C], f"IG {Prop_Name}")

#         # Write Input Properties
#         for C in range(InC):
#             Prop_Name = Input_Properties["Names"][C]
#             _Add_Scalar_Array(Image_Data, Input_Batch_Np[Bi, C], Prop_Name)

#         # Write Ground Truth and Predicted Cavern
#         _Add_Scalar_Array(Image_Data, Output_Batch_Np[Bi, 0], "Ground Truth Cavern")
#         _Add_Scalar_Array(Image_Data, Output_Batch_Pred_Np[Bi, 0], "Predicted Cavern")

#         # Write to Single .vti File
#         Filename = os.path.join(Vtk_Dir, f"{Sample_Name}.vti")
#         Writer = vtk.vtkXMLImageDataWriter()
#         Writer.SetFileName(Filename)
#         Writer.SetInputData(Image_Data)
#         Writer.Write()


def WriteResultsToCSV(Dataset: torch.utils.data.DataLoader, Model_List, Device: torch.device, Result_Path: str,
                      Split_Name: str, Ensemble_Toggle: bool, Metrics) -> None:
    """
    Summary: Runs the model (or ensemble of models) over Dataset, computes
    per-sample IoU/FScore/Accuracy, and writes a per-sample results CSV to
    Result_Path/{Split_Name}_Results.csv.
    """

    # Move Model(s) to Device
    if Ensemble_Toggle is True:
        Model_List = [Model.to(Device) for Model in Model_List]
        for Model in Model_List:
            Model.eval()
    else:
        Single_Model = Model_List[0] if isinstance(Model_List, list) else Model_List
        Single_Model = Single_Model.to(Device)
        Single_Model.eval()
        Model_List = Single_Model

    Metrics_Fn = Metrics

    # Declare Storage Arrays
    Dataset_Name_All = []
    Dataset_Type_All = []
    Backfill_All = []
    IoU_All = []
    FScore_All = []
    Accuracy_All = []
    Tonnage_Error_All = []

    with torch.no_grad():
        for Input_Batch, Output_Batch, Dataset_Name_Batch, _, Backfill_Batch, Dataset_Type_Batch, _, Density_Batch, Uranium_Batch in Dataset:

            # Modify Batch Shapes
            Input_Batch    = Input_Batch.to(Device, dtype=torch.float32, non_blocking=True)
            Output_Batch   = Output_Batch.to(Device, dtype=torch.float32, non_blocking=True)
            Backfill_Batch = Backfill_Batch.to(Device, dtype=torch.float32, non_blocking=True)
            Density_Batch  = Density_Batch.to(Device, dtype=torch.float32, non_blocking=True)
            Uranium_Batch  = Uranium_Batch.to(Device, dtype=torch.float32, non_blocking=True)

            # Model Prediction
            Output_Batch_Pred = torch.zeros_like(Output_Batch)
            if Ensemble_Toggle is True:
                for Model in Model_List:
                    Output_Batch_Pred += torch.sigmoid(Model(Input_Batch))
                Output_Batch_Pred = Output_Batch_Pred / len(Model_List)
            else:
                Output_Batch_Pred = torch.sigmoid(Model_List(Input_Batch))

            # Compute Metrics (per-sample)
            Metrics_Batch = Metrics_Fn(Output_Batch_Pred, Output_Batch, Density_Batch, Uranium_Batch, Input_Batch)

            # Stack Data
            Dataset_Name_All.extend(Dataset_Name_Batch)
            Dataset_Type_All.extend(Dataset_Type_Batch)
            Backfill_All.extend((Backfill_Batch.flatten(1).sum(dim=1) > 0).cpu().numpy())
            IoU_All.extend(Metrics_Batch['IoU'].cpu().numpy())
            FScore_All.extend(Metrics_Batch['FScore'].cpu().numpy())
            Accuracy_All.extend(Metrics_Batch['Accuracy'].cpu().numpy())
            Tonnage_Error_All.extend(Metrics_Batch['Tonnage_Error'].cpu().numpy())

    # Write to CSV
    Columns = {'Dataset Name':      Dataset_Name_All,
               'Backfilled Cavity': Backfill_All,
               'IoU [-]':           IoU_All,
               'FScore [-]':        FScore_All,
               'Accuracy [-]':      Accuracy_All,
               'Data Type':         Dataset_Type_All,
               'Tonnage Error [-]': Tonnage_Error_All}

    Df = pd.DataFrame(Columns)
    Df.to_csv(os.path.join(Result_Path, f"{Split_Name}_Results.csv"), index=False)