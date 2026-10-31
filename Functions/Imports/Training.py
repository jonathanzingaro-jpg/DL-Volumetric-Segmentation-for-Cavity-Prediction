"""
====================================================================
File: Functions/Imports.py
Description:
    Import Module Function for training
====================================================================
"""

# Libraries
import os
import sys
from pathlib import Path
from contextlib2 import redirect_stdout
import pandas as pd

# Third Party Libraries
import numpy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.io import savemat
from scipy import ndimage
import captum
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import sklearn
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from functools import partial
from torchinfo import summary
import torch.multiprocessing as mp
from typing import Any, Dict

# Plotting Libraries
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# Project Functions / Classes

# Pre-Processing Functions
from Functions.Preprocessing.DataImport import DataImport
from Functions.Preprocessing.DataNormalization import *

# Dataset Class
from Functions.Dataset.CavityDataset import DiskBatchDataset, BuildLoaders, Apply_GPU_Augmentation

# Training Functions
from Functions.Metrics.CustomMetrics import SegmentationMetrics, SegmentationLoss
from Functions.PostProcessing.ComputeResults import ComputeResults

# Custom Models
from Functions.Models.CustomModel import CustomCNN
from Functions.Models.UNet3D import UNet3D
from Functions.Models.InitializeModel import InitializeModel, InitializeOptimizer, InitializeScheduler

# Post Processing Functions
from Functions.PostProcessing.SHAPAnalysis import ShapAnalysis
from Functions.PostProcessing.WriteResults import WriteResultsToCSV
from Functions.PostProcessing.Plotting import PostProcessPlotting, PlotPrediction, Plot_Single_Dataset, PlotStatisticsCurve

# Helper Functions
from Functions.HelperFunctions import *

# Error Function
from Functions.ErrorFunctions import ValidateConfig, ConfigError, ValidateDatasetShape

# Public API
__all__ = [
    # Standard Library
    "os",
    "sys",
    "Path",
    "redirect_stdout",
    "pd",

    # Third Party Libraries
    "numpy",
    "tqdm",
    "torch",
    "nn",
    "F",
    "DataLoader",
    "Dataset",
    "savemat",
    "ndimage",
    "captum",
    "mp",
    "ProcessPoolExecutor",
    "sklearn",
    "roc_curve",
    "auc",
    "precision_recall_curve",
    "summary", 
    "partial",
    "mp",
    "Any",
    "Dict",

    # Plotting Libraries
    "matplotlib",
    "plt",
    "Normalize",
    "ScalarMappable",
    "Patch",
    "Line2D",
    "gridspec",

    # Pre-Processing Functions
    "DataImport",
    "DataNormalization",

    # Dataset Class
    "DiskBatchDataset",
    "BuildLoaders",
    "Apply_GPU_Augmentation",

    # Training Functions / Metrics
    "SegmentationMetrics", 
    "SegmentationLoss",
    "ComputeResults",

    # Custom Models
    "CustomCNN",
    "UNet3D",
    "InitializeModel",
    "InitializeOptimizer",
    "InitializeScheduler",

    # Post Processing Functions
    "ShapAnalysis",
    "WriteResultsToCSV",
    "PostProcessPlotting",
    "PlotPrediction",
    "Plot_Single_Dataset",
    "PlotStatisticsCurve",

    # Helper Functions
    "TS_Print",
    "DeclareDevice",

    # Error Function
    "ValidateConfig", 
    "ConfigError",
    "ValidateDatasetShape"
]