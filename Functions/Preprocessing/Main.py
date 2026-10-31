# Functions/Main.py

# Custom Functions
from Functions.HelperFunctions import *
from Functions.Preprocessing.DataImport import *
from Functions.Preprocessing.DataNormalization import *
from Functions.Dataset.CavityDataset import *
from typing import List, Tuple
import os

def Main(CurDir, cfg):

    # Extract Configuration Parameters
    Input_Keys: Tuple[str, ...]        = tuple(cfg["Input"].keys())
    Output_Keys: Tuple[str, ...]       = tuple(cfg["Output"].keys())
    Input_Norm_Type: Tuple[str, ...]   = tuple(cfg["Input"][k] for k in Input_Keys)
    Output_Norm_Type: Tuple[str, ...]  = tuple(cfg["Output"][k] for k in Output_Keys)
    Dataset_Split: List[float]         = cfg["Dataset_Split"]
    Split_Seed: int                    = cfg.get("Split_Seed", 42)
    Raw_Data_Name: str                 = cfg["Raw_Data_Name"]
    Dataset_Name: str                  = cfg["Dataset_Name"]
    Remove_Regions: bool               = cfg["Remove_Regions"]
    Problem_Type: str                  = cfg["Problem_Type"]
    CNN_Dim: int                       = cfg["CNN_Dim"]
    Batch_Size: int                    = cfg["Batch_Size"]

    # Create Paths
    DatasetDir = os.path.join(CurDir, 'Dataset', f'{Dataset_Name}')
    BatchDir = os.path.join(DatasetDir, 'Raw Batches')

    # Data Import (Save Raw Data into Batches, split per-cavity into Train/Validation/Test)
    TS_Print("Starting Data Import")
    MetaData = DataImport(CurDir, Input_Keys, Output_Keys, Raw_Data_Name, Dataset_Name,
                           Remove_Regions, Problem_Type, CNN_Dim, Batch_Size,
                           Split_Ratios=tuple(Dataset_Split), Split_Seed=Split_Seed)
    TS_Print("Data Import Completed")

    # Normalize Data
    TS_Print("Starting Data Normalization")

    # Declare Normalization Class
    Norm = DataNormalization(Input_Norm_Type, Input_Keys,
                              Output_Norm_Type, Output_Keys, MetaData)

    # Fit and Save Normalization Parameters (Fit on Train Split)
    Norm.Fit(BatchDir=os.path.join(BatchDir, 'Train'))
    Norm.Save(CurDir, Dataset_Name)

    # Normalize All Splits (Writes Processed Batches/{Train,Validation,Test})
    Norm.Process(DatasetDir, BatchDir)
    TS_Print("Data Normalization Completed")