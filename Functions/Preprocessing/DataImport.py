# Functions/Preprocessing/DataImport.py

# Import Libraries 
import os, torch, numpy, random
from scipy import ndimage
from pathlib import Path 
import re

# Custom Functions
from Functions.HelperFunctions import * 

## Data Import Function
def DataImport(curDir: str, Input_Keys: list, Output_Keys: list, Raw_Data_Name: str, 
               Dataset_Name: str, Remove_Regions: bool, Problem_Type: str, CNN_Dim: tuple, Batch_Size: int,
               Split_Ratios: tuple, Split_Seed: int = 42) -> dict:
    """
    Load cavity tensor data and save it to disk in batches of .npz files, grouped by
    Train / Validation / Test split.

    Problem_Type controls how each cavity's 'inc_XXX' files are turned into samples:

      - 'Static':       CNN_Dim[-1] must be 1. Each cavity contributes exactly ONE sample
                         (it may only have a single 'inc_XXX' file).
      - 'Incremental':  CNN_Dim[-1] must be 1. Each cavity's 'inc_XXX' files are EACH turned
                         into their own sample (Nt = 1 per sample), so a cavity with 5
                         timesteps contributes 5 samples to N. Cavities still stay grouped
                         within a single Train/Validation/Test split -- split assignment is
                         by cavity, not by file.
      - 'Dynamic':      CNN_Dim[-1] must be > 1. Each cavity contributes exactly ONE sample,
                         assembled as (*CNN_Dim, keys), where each 'inc_XXX' file fills one
                         temporal slot (axis -2) of that sample's tensor.

    Cavity NAMES are shuffled and then randomly assigned to Train/Validation/Test according
    to Split_Ratios. All files/samples belonging to a given cavity stay together in the same 
    split, AND in the same saved batch file.
    """

    # Splits Definition
    Split_Names = ('Train', 'Validation', 'Test')

    if len(Split_Ratios) != 3:
        raise ValueError(f"Split_Ratios must have 3 values (Train, Validation, Test), got {Split_Ratios}")
    if abs(sum(Split_Ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split_Ratios must sum to 1.0, got {sum(Split_Ratios)} ({Split_Ratios})")
    if any(r < 0 for r in Split_Ratios):
        raise ValueError(f"Split_Ratios must be non-negative, got {Split_Ratios}")

    # Create and Define Dataset Directory
    DatasetDir = os.path.join(curDir, 'Dataset', f'{Dataset_Name}')
    BatchDir = os.path.join(DatasetDir, 'Raw Batches')
    DataDir = os.path.join(curDir, 'Raw Data', Raw_Data_Name)
    os.makedirs(DatasetDir, exist_ok=True)

    Split_Dirs = {}
    for Split in Split_Names:
        Split_Dirs[Split] = os.path.join(BatchDir, Split)
        os.makedirs(Split_Dirs[Split], exist_ok=True)

    # Raise Errors
    if not os.path.isdir(DataDir):
        raise FileNotFoundError(f"Raw data directory not found: {DataDir}")

    if Problem_Type not in ("Static", "Incremental", "Dynamic"):
        raise ValueError(f"{Problem_Type} is not supported. Supported options are ['Static', 'Incremental', 'Dynamic']")

    if Problem_Type in ("Static", "Incremental") and CNN_Dim[-1] != 1:
        raise ValueError(
            f"'{Problem_Type}' problem type requires a temporal padding dim of 1 (each sample is "
            f"single-timestep), but got a dimension of {CNN_Dim[-1]}."
        )

    if Problem_Type == "Dynamic" and CNN_Dim[-1] == 1:
        raise ValueError(f"Dynamic problem type requires temporal padding > 1, but got a dimension of {CNN_Dim[-1]}.")

    # Retrieve Dataset Paths
    Cavity_Names = [d for d in os.listdir(DataDir) if os.path.isdir(os.path.join(DataDir, d)) and not d.startswith('.')]

    # Shuffle Cavity Names, then Randomly Assign Each to a Split
    Shuffled_Cavities = Cavity_Names.copy()
    random.Random(Split_Seed).shuffle(Shuffled_Cavities)

    N_Total = len(Shuffled_Cavities)
    N_Train = round(Split_Ratios[0] * N_Total)
    N_Val = round(Split_Ratios[1] * N_Total)
    N_Test = N_Total - N_Train - N_Val

    Cavity_Split = {}
    for Name in Shuffled_Cavities[:N_Train]:
        Cavity_Split[Name] = 'Train'
    for Name in Shuffled_Cavities[N_Train:N_Train + N_Val]:
        Cavity_Split[Name] = 'Validation'
    for Name in Shuffled_Cavities[N_Train + N_Val:]:
        Cavity_Split[Name] = 'Test'

    TS_Print(f"Split sizes -> Train: {N_Train}, Validation: {N_Val}, Test: {N_Test} (of {N_Total} cavities, seed={Split_Seed})")

    for Split in Split_Names:
        if Split_Ratios[Split_Names.index(Split)] > 0 and sum(1 for v in Cavity_Split.values() if v == Split) == 0:
            TS_Print(f"WARNING: Split '{Split}' has a non-zero ratio but received 0 cavities (too few cavities for these ratios).")

    # Tracking Across All Batches / Splits
    Multi_Region = []
    Geology_Idx = None
    Backfill_Idx_Meta = None

    # Per-Split Tracking
    Batch_Idx = {Split: 0 for Split in Split_Names}
    Sample_Count = {Split: 0 for Split in Split_Names}
    Batch_Fill = {Split: 0 for Split in Split_Names}    
    Cavity_Fill = {Split: 0 for Split in Split_Names}     

    # Per-Split Batch Accumulators (reset after each save)
    Input_List = {Split: [] for Split in Split_Names}
    Output_List = {Split: [] for Split in Split_Names}
    Coord_List = {Split: [] for Split in Split_Names}
    Backfill_List = {Split: [] for Split in Split_Names}
    Cavity_Name_List = {Split: [] for Split in Split_Names}
    Dataset_Type_List = {Split: [] for Split in Split_Names}
    Density_List = {Split: [] for Split in Split_Names}
    Uranium_List = {Split: [] for Split in Split_Names}
    Multi_Region_List = {Split: [] for Split in Split_Names}     

    def Save_Batch(Split):
        nonlocal Batch_Idx, Batch_Fill

        Input = numpy.stack(Input_List[Split], axis=0)
        Output = numpy.stack(Output_List[Split], axis=0)
        Backfill = torch.stack(Backfill_List[Split], dim=0)
        Coord = torch.stack(Coord_List[Split], dim=0)
        Density = torch.stack(Density_List[Split], dim=0)
        Uranium = torch.stack(Uranium_List[Split], dim=0)

        Channel_Dim = Input.shape[-2]
        if Problem_Type in ("Static", "Incremental") and Channel_Dim != 1:
            raise ValueError(
                f"'{Problem_Type}' problem expects channel dim 1, got {Channel_Dim} "
                f"(full tensor shape {tuple(Input.shape)})"
            )

        Batch_Path = os.path.join(Split_Dirs[Split], f"batch_{Batch_Idx[Split]:04d}.npz")
        numpy.savez_compressed(
            Batch_Path,
            Input=Input,
            Output=Output,
            Backfill=Backfill.numpy(),
            Coord=Coord.numpy(),
            Density=Density.numpy(),
            Uranium=Uranium.numpy(),
            Cavity_Name=numpy.array(Cavity_Name_List[Split]),
            Dataset_Type=numpy.array(Dataset_Type_List[Split]),
            Multi_Region=numpy.array(Multi_Region_List[Split]),
        )

        TS_Print(f"[{Split}] Saved batch {Batch_Idx[Split]} ({Batch_Fill[Split]} samples, Input Shape {tuple(Input.shape)}) -> {Batch_Path}")
        Batch_Idx[Split] += 1

        Input_List[Split].clear(); Output_List[Split].clear(); Coord_List[Split].clear(); Backfill_List[Split].clear()
        Cavity_Name_List[Split].clear(); Dataset_Type_List[Split].clear(); Density_List[Split].clear(); Uranium_List[Split].clear()
        Multi_Region_List[Split].clear()
        Batch_Fill[Split] = 0

    def Load_File(File, Cavity_Name):
        """
        Load a single 'inc_XXX' cavity tensor file and return everything needed to create the sample.
        """
        nonlocal Geology_Idx, Backfill_Idx_Meta

        with numpy.load(File, allow_pickle=True) as npz:

            if 'data' not in npz or 'columns' not in npz:
                raise KeyError(f"'{Cavity_Name}' tensor file missing 'data' or 'columns' array: {File}")

            Data_Tensor = npz['data']
            Data_Keys = list(npz['columns'])
            Data_Keys_Type = list(npz['column_types'])

            # Extract Geology Indices for Inputs
            if Geology_Idx is None:
                Geology_Idx_All = [i for i, x in enumerate(Data_Keys_Type) if x == 'Geology']
                Geology_Values = [Data_Keys[i] for i in Geology_Idx_All]
                Geology_Property_Names = list(set(Geology_Values) & set(Input_Keys))
                Geology_Idx = [Input_Keys.index(k) for k in Geology_Property_Names]

            # Clean up Cavity Shape
            Cavity_Key = next((k for k in ('consumed_cavity', 'next_inc_cavity') if k in Data_Keys), None)

            File_Is_Multi_Region = False
            if Cavity_Key is not None and Remove_Regions is True:
                Consumed_Cavity_Idx = Data_Keys.index(Cavity_Key)
                Output_Labels, n = ndimage.label(Data_Tensor[..., Consumed_Cavity_Idx])
                Sizes = numpy.bincount(Output_Labels.ravel())[1:]
                Keep = 1 + numpy.argmax(Sizes)
                Data_Tensor[..., Consumed_Cavity_Idx] = (Output_Labels == Keep)
                if n > 1:
                    File_Is_Multi_Region = True
                    if Cavity_Name not in Multi_Region:
                        Multi_Region.append(Cavity_Name)

            # Convert Data_Tensor to Pytorch
            Data_Tensor = torch.from_numpy(Data_Tensor)

            # Map Requested Keys to Indices
            try:
                Input_Idx = [Data_Keys.index(k) for k in Input_Keys]
                Output_Idx = [Data_Keys.index(k) for k in Output_Keys]
                Backfill_Idx = Data_Keys.index('backfill_density')
                Coord_Keys = ['centroid_x', 'centroid_y', 'centroid_z']
                Coord_Idx = [Data_Keys.index(k) for k in Coord_Keys]
                Density_Idx = Data_Keys.index('den')
                Uranium_Idx = Data_Keys.index('u3o8_pct')
            except ValueError as e:
                raise ValueError(f"Key not found in columns for {Cavity_Name}: {e}. Available columns: {Data_Keys}")

            if Backfill_Idx_Meta is None:
                Backfill_Idx_Meta = Input_Keys.index('backfill_density')

            Input_Tensor = Data_Tensor[..., Input_Idx]
            Output_Tensor = Data_Tensor[..., Output_Idx]

            # Spatial Dimenisons of Tensor
            Spatial_Target = CNN_Dim[:-1]
            Actual_Shape = Input_Tensor.shape[:-1]
            for dim, (Actual, Target) in enumerate(zip(Actual_Shape, Spatial_Target)):
                if Actual > Target:
                    raise ValueError(
                        f"'{Cavity_Name}' file {File.name} spatial dim {dim} ({Actual}) exceeds "
                        f"CNN_Dim ({Target}); cannot fit into pre-allocated tensor."
                    )

            # Write NaN Density with Mean
            Density = Data_Tensor[..., Density_Idx]
            Density_Mean = torch.nanmean(Density)
            Density = torch.nan_to_num(Density, nan=Density_Mean.item())

            # Write NaN Uranium with Mean
            Uranium = Data_Tensor[..., Uranium_Idx]
            Uranium_Mean = torch.nanmean(Uranium)
            Uranium = torch.nan_to_num(Uranium, nan=Uranium_Mean.item())

            Coord = Data_Tensor[..., Coord_Idx]
            Backfill = Data_Tensor[..., Backfill_Idx]
            Dataset_Type_Label = 'Synthetic' if 'synth' in Cavity_Name else 'Field'

            return {
                'Input': Input_Tensor.numpy(),
                'Output': Output_Tensor.numpy(),
                'Coord': Coord,
                'Backfill': Backfill,
                'Density': Density,
                'Uranium': Uranium,
                'Dataset_Type': Dataset_Type_Label,
                'Is_Multi_Region': File_Is_Multi_Region,
            }

    def Append_Sample(Split, Cavity_Name, Input_Arr, Output_Arr, Coord, Backfill, Density, Uranium,
                       Dataset_Type_Label, Is_Multi_Region):
        """Append single cavity data to accumulations lists with the entire batch."""

        Input_List[Split].append(Input_Arr)
        Output_List[Split].append(Output_Arr)
        Cavity_Name_List[Split].append(Cavity_Name)
        Dataset_Type_List[Split].append(Dataset_Type_Label)
        Coord_List[Split].append(Coord)
        Backfill_List[Split].append(Backfill)
        Density_List[Split].append(Density)
        Uranium_List[Split].append(Uranium)
        Multi_Region_List[Split].append(Is_Multi_Region)

        Batch_Fill[Split] += 1
        Sample_Count[Split] += 1

    # Loop Dataset
    for Cavity_Name in Cavity_Names:

        Split = Cavity_Split[Cavity_Name]

        def sort_key(f):
            m = re.search(r"inc_(\d+)", f.name)
            return int(m.group(1)) if m else 0

        Files = sorted(Path(os.path.join(DataDir, Cavity_Name)).rglob("*_cavity_tensor.npz"), key=sort_key)

        if Problem_Type in ("Static", "Dynamic"):

            # Check Number of Files Fits within the Temporal Padding Dim 
            if len(Files) > CNN_Dim[-1]:
                raise ValueError(
                    f"'{Cavity_Name}' has {len(Files)} temporal files, exceeding CNN_Dim temporal size ({CNN_Dim[-1]})."
                )

            Input_Tensor_Single = numpy.zeros((*CNN_Dim, len(Input_Keys)), dtype=numpy.float32)
            Output_Tensor_Single = numpy.zeros((*CNN_Dim, len(Output_Keys)), dtype=numpy.float32)
            Cavity_Is_Multi_Region = False
            Last_Coord = Last_Backfill = Last_Density = Last_Uranium = Last_Dataset_Type = None

            for t, File in enumerate(Files):
                Loaded = Load_File(File, Cavity_Name)

                Input_Tensor_Single[..., (t,), :] = Loaded['Input']
                Output_Tensor_Single[..., (t,), :] = Loaded['Output']

                Cavity_Is_Multi_Region = Cavity_Is_Multi_Region or Loaded['Is_Multi_Region']
                Last_Coord = Loaded['Coord']
                Last_Backfill = Loaded['Backfill']
                Last_Density = Loaded['Density']
                Last_Uranium = Loaded['Uranium']
                Last_Dataset_Type = Loaded['Dataset_Type']

            Append_Sample(Split, Cavity_Name, Input_Tensor_Single, Output_Tensor_Single,
                          Last_Coord, Last_Backfill, Last_Density, Last_Uranium,
                          Last_Dataset_Type, Cavity_Is_Multi_Region)

        else:   # Incremental Import -> Flattens Temporal to Seperate Input-Output Pairs

            for t, File in enumerate(Files):
                Loaded = Load_File(File, Cavity_Name)

                Inc_Match = re.search(r"inc_(\d+)", File.name)
                Inc_Num = int(Inc_Match.group(1)) if Inc_Match else t
                Sample_Name = f"{Cavity_Name}_inc_{Inc_Num:03d}"

                Input_Tensor_Single = numpy.zeros((*CNN_Dim, len(Input_Keys)), dtype=numpy.float32)
                Output_Tensor_Single = numpy.zeros((*CNN_Dim, len(Output_Keys)), dtype=numpy.float32)
                Input_Tensor_Single[..., (0,), :] = Loaded['Input']
                Output_Tensor_Single[..., (0,), :] = Loaded['Output']

                Append_Sample(Split, Sample_Name, Input_Tensor_Single, Output_Tensor_Single,
                              Loaded['Coord'], Loaded['Backfill'], Loaded['Density'], Loaded['Uranium'],
                              Loaded['Dataset_Type'], Loaded['Is_Multi_Region'])

        # Save Full Batch
        Cavity_Fill[Split] += 1
        if Cavity_Fill[Split] == Batch_Size:
            TS_Print(f"[{Split}] Saving batch {Batch_Idx[Split]} ({Cavity_Fill[Split]} cavities, {Batch_Fill[Split]} samples)...")
            Save_Batch(Split)
            Cavity_Fill[Split] = 0

    # Save Partial Batches
    for Split in Split_Names:
        if Batch_Fill[Split] > 0:
            Save_Batch(Split)
            Cavity_Fill[Split] = 0

    Total_Samples = sum(Sample_Count.values())
    Total_Batches = sum(Batch_Idx.values())
    TS_Print(f"Total Samples: {Total_Samples} across {Total_Batches} batches")
    for Split in Split_Names:
        TS_Print(f"  {Split}: {Sample_Count[Split]} samples, {Batch_Idx[Split]} batches -> {Split_Dirs[Split]}")
    TS_Print(f"Multi-Region Cavities: {Multi_Region}")

    return {'Geology_Idx': Geology_Idx,
            'Backfill_Idx': Backfill_Idx_Meta,
            'Multi_Region': Multi_Region,
            'Cavity_Split': Cavity_Split,}