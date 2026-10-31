"""
====================================================================
File: RunTraining.py

Description:
Trains the neural network that predicts voxelized cavern shapes from
preprocessed geological input properties.

This script runs training for the static dataset configuration only.
Note that the broader model framework also supports an ensembled model,
as well as post-processing steps such as plotting predictions on the
test dataset.

The configuration file referenced by this script points to the
Field_Static_321_V1 dataset. See the project documentation for further details
on dataset generation and model architecture.

See JBS Blocks Consumed Model Reference Manual (v1.1.2) for more detail.

Author:       Giovanni Zingaro
Supervisor:   Dr. Robert Gracie, Dr. John Quilty
Organization: University of Waterloo
Created: 2026-06-14
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
from Functions.Training.Main import Main

## Load Config
CONFIG_PATH = PROJECT_ROOT / "Config" / "Training" / "Config_Field_Static_V1.json"
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

## Run Training
if __name__ == "__main__":   

    print('Start Training ...')
    Main(curDir, cfg)
    print('Complete!')