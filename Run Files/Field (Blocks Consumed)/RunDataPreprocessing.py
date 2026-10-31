"""
====================================================================
File: RunDataPreprocessing.py
Description:
    Preprocesses the input and output tensors required to train a 
    CNN-based model. Supports log normalization and min–max scaling 
    and optional selection of input features. See JBS Blocks Consumed 
    Model Reference Manual (v1.1.2) for more detail.
Author:  Giovanni Zingaro
Created: 2025-10-15
Updated: 2025-12-08
====================================================================
"""

## Import Libraries
import sys
import json
from pathlib import Path

## Set Current Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
curDir = str(PROJECT_ROOT)

## Import Main File
from Functions.Preprocessing.Main import Main

## Load Config
CONFIG_PATH = PROJECT_ROOT / "Config" / "Data Preprocessing" / "Config_Field_Static_V1.json"
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

if __name__ == "__main__":
    
    Main(curDir, cfg)
