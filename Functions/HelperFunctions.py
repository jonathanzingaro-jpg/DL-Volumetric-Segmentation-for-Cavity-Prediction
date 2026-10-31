# Functions/HelperFunctions

# Import Libraries
from datetime import datetime
import torch

def TS_Print(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts}: {msg}")

def DeclareDevice():
    """
    Initial Device Class for Pytorch -> "CUDA", "MPS" and "CPU"
    """
    # Initialize Device
    if torch.cuda.is_available():
        Device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        Device = torch.device("mps")
    else:
        Device = torch.device("cpu")
    return Device