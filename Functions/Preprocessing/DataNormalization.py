# Functions/DataNormalization.py

# Import Libraries 
import torch, numpy, os
from torch.utils.data import Dataset 
from pathlib import Path

# Custom Functions
from Functions.HelperFunctions import * 

class DataNormalization():
    def __init__(self, Input_Norm_Type: tuple, Input_Keys: list, Output_Norm_Type: tuple, Output_Keys: list, MetaData: dict):

        self.Input_Norm_Type = Input_Norm_Type                             # Type of Normalization for Input Keys
        self.Output_Norm_Type = Output_Norm_Type                           # Type of Normalization for Output Keys
        self.Input_Keys = Input_Keys                                       # Input Keys
        self.Output_Keys = Output_Keys                                     # Output Keys
        self.Params = {}                                                   # Empty Data Normalization Structure
        self.Eps = 1e-6                                                    # Floor for MinMax Scaling
        self.Idx = {'Geology_Idx': MetaData['Geology_Idx'],                # Save Indices
                    'Backfill_Idx': MetaData['Backfill_Idx']}             

    def Fit(self, BatchDir: str) -> None:
        """
        Compute per-channel min/max stats, ignoring NaN (null) values, by looping
        over saved batch files rather than requiring the full dataset in memory.
        Assumes each batch's Input shape (N, X, Y, Z, T, C) and Output shape (N, X, Y, Z, C_out).
        """

        if self.Params:
            raise RuntimeError("self.Params is already populated. Create a new instance or call Load() instead of Fit().")

        Batch_Files = sorted(Path(BatchDir).glob("batch_*.npz"))
        if not Batch_Files:
            raise FileNotFoundError(f"No batch files found in {BatchDir}")

        Input_Max = None
        Input_Min = None
        Output_Max = None
        Output_Min = None

        # Channel Reduction Dims
        Input_Reduce_Dims = (0, 1, 2, 3, 4)
        Output_Reduce_Dims = (0, 1, 2, 3)

        for Batch_File in Batch_Files:

            with numpy.load(Batch_File, allow_pickle=True) as npz:
                Input = torch.from_numpy(npz['Input']).clone()
                Output = torch.from_numpy(npz['Output']).clone()

            Null_Mask_Input = torch.isnan(Input)
            Null_Mask_Output = torch.isnan(Output)

            # Logarithmic Normalization
            if "Log" in self.Input_Norm_Type:
                Idx = [i for i, k in enumerate(self.Input_Norm_Type) if k == "Log"]
                Input[..., Idx] = torch.log1p(Input[..., Idx])

            if "Log" in self.Output_Norm_Type:
                Idx = [i for i, k in enumerate(self.Output_Norm_Type) if k == "Log"]
                Output[..., Idx] = torch.log1p(Output[..., Idx])

            # Input Max/Min for this Batch
            Input_For_Max = torch.where(Null_Mask_Input, torch.tensor(float('-inf')), Input)
            Batch_Input_Max = torch.amax(Input_For_Max, dim=Input_Reduce_Dims, keepdim=True)
            Input_For_Min = torch.where(Null_Mask_Input, torch.tensor(float('inf')), Input)
            Batch_Input_Min = torch.amin(Input_For_Min, dim=Input_Reduce_Dims, keepdim=True)

            # Output Max/Min for this Batch
            Output_For_Max = torch.where(Null_Mask_Output, torch.tensor(float('-inf')), Output)
            Batch_Output_Max = torch.amax(Output_For_Max, dim=Output_Reduce_Dims, keepdim=True)
            Output_For_Min = torch.where(Null_Mask_Output, torch.tensor(float('inf')), Output)
            Batch_Output_Min = torch.amin(Output_For_Min, dim=Output_Reduce_Dims, keepdim=True)

            # Fold into Running Min/Max Across All Batches
            Input_Max = Batch_Input_Max if Input_Max is None else torch.maximum(Input_Max, Batch_Input_Max)
            Input_Min = Batch_Input_Min if Input_Min is None else torch.minimum(Input_Min, Batch_Input_Min)
            Output_Max = Batch_Output_Max if Output_Max is None else torch.maximum(Output_Max, Batch_Output_Max)
            Output_Min = Batch_Output_Min if Output_Min is None else torch.minimum(Output_Min, Batch_Output_Min)

        self.Params['Input']      = {'min': Input_Min.clone(), 'max': Input_Max.clone()}
        self.Params['Output']     = {'min': Output_Min.clone(), 'max': Output_Max.clone()}
        self.Params['Keys']       = {'Input_Keys': self.Input_Keys, 'Output_Keys': self.Output_Keys}
        self.Params['Norm_Type']  = {'Input_Norm_Type': self.Input_Norm_Type, 'Output_Norm_Type': self.Output_Norm_Type}
        self.Params['Idx']        = {'Geology_Idx': self.Idx['Geology_Idx'], 'Backfill_Idx': self.Idx['Backfill_Idx']}

        # Print as a Key | Min | Max table
        Input_Min_List = Input_Min.squeeze().tolist()
        Input_Max_List = Input_Max.squeeze().tolist()
        Output_Min_List = Output_Min.squeeze().tolist()
        Output_Max_List = Output_Max.squeeze().tolist()

        # Normalize to lists
        if not isinstance(Input_Min_List, list):
            Input_Min_List = [Input_Min_List]
        if not isinstance(Input_Max_List, list):
            Input_Max_List = [Input_Max_List]
        if not isinstance(Output_Min_List, list):
            Output_Min_List = [Output_Min_List]
        if not isinstance(Output_Max_List, list):
            Output_Max_List = [Output_Max_List]

        Key_Width = max(len(k) for k in self.Input_Keys + self.Output_Keys) + 2

        print(f"\n{'Key':<{Key_Width}}{'Min':>12}{'Max':>12}")
        print("-" * (Key_Width + 24))
        for Key, Min_Val, Max_Val, Norm in zip(self.Input_Keys, Input_Min_List, Input_Max_List, self.Input_Norm_Type):
            if Norm == "Log":
                Min_Val, Max_Val = float(numpy.exp(Min_Val)), float(numpy.exp(Max_Val))
            print(f"{Key:<{Key_Width}}{Min_Val:>12.4f}{Max_Val:>12.4f}")
        for Key, Min_Val, Max_Val, Norm in zip(self.Output_Keys, Output_Min_List, Output_Max_List, self.Output_Norm_Type):
            if Norm == "Log":
                Min_Val, Max_Val = float(numpy.exp(Min_Val)), float(numpy.exp(Max_Val))
            print(f"{Key:<{Key_Width}}{Min_Val:>12.4f}{Max_Val:>12.4f}")
        print('\n')

    def Save(self, CurDir: str, Dataset_Name: str) -> str:
        """Save self.Params to params.pt for later inference/reuse."""

        if not self.Params:
            raise RuntimeError("No params to save. Call Fit() first.")

        Save_Path = os.path.join(CurDir, 'Dataset', f'{Dataset_Name}', 'params.pt')
        os.makedirs(os.path.dirname(Save_Path), exist_ok=True)
        torch.save(self.Params, Save_Path)
        TS_Print(f"Saved Normalization Params to: {Save_Path}")
        return Save_Path

    def Load(self, CurDir: str, Dataset_Name: str) -> None:
        """Load previously-fit normalization params from disk."""

        Load_Path = os.path.join(CurDir, 'Dataset', f'{Dataset_Name}', 'params.pt')
        self.Params = torch.load(Load_Path)
        TS_Print(f"Loaded Normalization Params from: {Load_Path}")

    def Transform(self, Input: torch.Tensor, Output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply stored per-channel min-max params to (new) Input/Output tensors [0,1].
        """
        if not self.Params:
            raise RuntimeError("No fitted params found. Call Fit() or Load() first.")

        Input = Input.clone()
        Output = Output.clone()

        # Record Null Values
        Null_Mask_Input = torch.isnan(Input)
        Null_Mask_Output = torch.isnan(Output)

        # Logrithmic Normalization 
        if "Log" in self.Input_Norm_Type:
            Idx = [i for i, k in enumerate(self.Input_Norm_Type) if k == "Log"]
            Input[..., Idx] = torch.log1p(Input[..., Idx])

        if "Log" in self.Output_Norm_Type:
            Idx = [i for i, k in enumerate(self.Output_Norm_Type) if k == "Log"]
            Output[..., Idx] = torch.log1p(Output[..., Idx])

        # MinMax, Scaled into [Eps, 1]
        Input_Min = self.Params['Input']['min']
        Input_Max = self.Params['Input']['max']
        Input_Range = Input_Max - Input_Min
        Output_Min = self.Params['Output']['min']
        Output_Max = self.Params['Output']['max']
        Output_Range = Output_Max - Output_Min

        if (Input_Range == 0).any() or (Output_Range == 0).any():
            raise ValueError("Zero-range channel detected (min == max); cannot MinMax-scale.")

        Input =  (Input - Input_Min) / Input_Range
        Output = (Output - Output_Min) / Output_Range

        # Restore Null Values as 0.0
        Input[Null_Mask_Input] = 0.0
        Output[Null_Mask_Output] = 0.0

        return Input, Output

    def Inverse(self, Input: torch.Tensor, Output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply inverse normalization using stored per-channel min-max params to
        Input/Output tensors.
        """
        if not self.Params:
            raise RuntimeError("No fitted params found. Call Fit() or Load() first.")

        Input = Input.clone()
        Output = Output.clone()

        # Identify Null Positions Before Inverse-Scaling
        Null_Mask_Input = (Input == 0.0)
        Null_Mask_Output = (Output == 0.0)

        # Inverse MinMax Scaling
        Input_Min = self.Params['Input']['min']
        Input_Max = self.Params['Input']['max']
        Input_Range = Input_Max - Input_Min
        Output_Min = self.Params['Output']['min']
        Output_Max = self.Params['Output']['max']
        Output_Range = Output_Max - Output_Min

        Input = Input / Input_Range + Input_Min
        Output = Output /  Output_Range + Output_Min

        # Inverse Logrithmic Normalziation 
        if "Log" in self.Input_Norm_Type:
            Idx = [i for i, k in enumerate(self.Input_Norm_Type) if k == "Log"]
            Input[..., Idx] = torch.expm1(Input[..., Idx])

        if "Log" in self.Output_Norm_Type:
            Idx = [i for i, k in enumerate(self.Output_Norm_Type) if k == "Log"]
            Output[..., Idx] = torch.expm1(Output[..., Idx])

        # Restore Null Values
        Input[Null_Mask_Input] = 0.0
        Output[Null_Mask_Output] = 0.0

        return Input, Output

    def Process(self, DatasetDir: str, BatchDir: str = None) -> str:
        """
        Loop over saved raw batch (.npz) files -- per Train/Validation/Test split
        subdirectory -- apply Transform() to each batch's Input/Output, and save
        the result as .pt files into DatasetDir/'Processed Batches'/<Split>.
        """
        if not self.Params:
            raise RuntimeError("No fitted params found. Call Fit() or Load() first.")

        if BatchDir is None:
            BatchDir = os.path.join(DatasetDir, 'Raw Batches')

        Split_Names = ('Train', 'Validation', 'Test')
        ProcessedDir = os.path.join(DatasetDir, 'Processed Batches')

        Any_Batches_Found = False

        for Split in Split_Names:

            Split_Raw_Dir = os.path.join(BatchDir, Split)
            Split_Processed_Dir = os.path.join(ProcessedDir, Split)

            if not os.path.isdir(Split_Raw_Dir):
                TS_Print(f"WARNING: No raw batch directory for split '{Split}' ({Split_Raw_Dir}); skipping.")
                continue

            Batch_Files = sorted(Path(Split_Raw_Dir).glob("batch_*.npz"))
            if not Batch_Files:
                TS_Print(f"WARNING: No batch files found for split '{Split}' in {Split_Raw_Dir}; skipping.")
                continue

            os.makedirs(Split_Processed_Dir, exist_ok=True)

            for Batch_File in Batch_Files:

                with numpy.load(Batch_File, allow_pickle=True) as npz:
                    Input = torch.from_numpy(npz['Input']).clone()
                    Output = torch.from_numpy(npz['Output']).clone()
                    Backfill = torch.from_numpy(npz['Backfill']).clone()
                    Coord = torch.from_numpy(npz['Coord']).clone()
                    Density = torch.from_numpy(npz['Density']).clone()
                    Uranium = torch.from_numpy(npz['Uranium']).clone()
                    Cavity_Name = npz['Cavity_Name'].tolist()
                    Dataset_Type = npz['Dataset_Type'].tolist()
                    Multi_Region = npz['Multi_Region'].tolist()

                Input_T, Output_T = self.Transform(Input, Output)

                Processed_Path = os.path.join(Split_Processed_Dir, Batch_File.stem + ".pt")
                torch.save({
                    'Input': Input_T,
                    'Output': Output_T,
                    'Backfill': Backfill,
                    'Coord': Coord,
                    'Density': Density,
                    'Uranium': Uranium,
                    'Cavity_Name': Cavity_Name,
                    'Dataset_Type': Dataset_Type,
                    'Multi_Region': Multi_Region,
                }, Processed_Path)

                Any_Batches_Found = True
                TS_Print(f"[{Split}] Processed {Batch_File.name} -> {Processed_Path}")

        if not Any_Batches_Found:
            raise FileNotFoundError(f"No batch files found in any split subdirectory of {BatchDir}")