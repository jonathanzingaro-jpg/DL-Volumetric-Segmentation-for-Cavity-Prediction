# Functions/Training/TemperatureScaling.py

# Import Libraries
import torch


def _Get_Target(Output_Batch: torch.Tensor, Input_Batch: torch.Tensor,
                Problem_Type: str, Previous_State_Idx: int) -> torch.Tensor:

    if Problem_Type != "Incremental":
        return Output_Batch
    Previous_State = Input_Batch[..., Previous_State_Idx].unsqueeze(-1)
    return Output_Batch - Previous_State


## Compute Expected Calibration Error
def Compute_ECE(Logits: torch.Tensor, Targets: torch.Tensor, N_Bins: int = 20) -> float:

    # Guard Against Empty Slices
    if Logits.numel() == 0:
        return float('nan')

    with torch.no_grad():

        # Confidence is Distance from the Decision Boundary, Not Raw Probability
        Prob    = torch.sigmoid(Logits)
        Pred    = (Prob > 0.5).float()
        Conf    = torch.where(Pred == 1, Prob, 1 - Prob)
        Correct = Pred.eq((Targets > 0.5).float()).float()

        # Bin by Confidence, Accumulate |Accuracy - Confidence| Weighted by Population
        Error = torch.zeros(1, device=Logits.device)
        Edges = torch.linspace(0.5, 1.0, N_Bins + 1, device=Logits.device)
        for Low, High in zip(Edges[:-1], Edges[1:]):
            Bin      = (Conf > Low) & (Conf <= High)
            Fraction = Bin.float().mean()
            if Fraction > 0:
                Error += (Correct[Bin].mean() - Conf[Bin].mean()).abs() * Fraction

        return Error.item()

## Compute Expected Calibration Error over Predicted-Positive Voxels
def Compute_ECE_Positive(Logits: torch.Tensor, Targets: torch.Tensor, N_Bins: int = 20) -> float:

    # Guard Against Empty Slices
    if Logits.numel() == 0:
        return float('nan')

    with torch.no_grad():

        Prob   = torch.sigmoid(Logits)
        Actual = (Targets > 0.5).float()

        # Select on PREDICTIONS
        Select = Prob > 0.5
        if Select.sum() == 0:
            return float('nan')
        Prob, Actual = Prob[Select], Actual[Select]

        # Bin by Predicted Probability, Accumulate |Observed Frequency - Probability|
        Error = torch.zeros(1, device=Logits.device)
        Edges = torch.linspace(0.5, 1.0, N_Bins + 1, device=Logits.device)
        for Low, High in zip(Edges[:-1], Edges[1:]):
            Bin      = (Prob > Low) & (Prob <= High)
            Fraction = Bin.float().mean()
            if Fraction > 0:
                Error += (Actual[Bin].mean() - Prob[Bin].mean()).abs() * Fraction

        return Error.item()

## Accumulate Bin Statistics for Expected Calibration Error
def Accumulate_Bin_Stats(Logits: torch.Tensor, Targets: torch.Tensor,
                          Bin_Correct: torch.Tensor, Bin_Conf: torch.Tensor,
                          Bin_Count: torch.Tensor, Edges: torch.Tensor) -> None:

    with torch.no_grad():

        # Confidence is Distance from the Decision Boundary, Not Raw Probability
        Prob    = torch.sigmoid(Logits).reshape(-1)
        Pred    = (Prob > 0.5).float()
        Conf    = torch.where(Pred == 1, Prob, 1 - Prob)
        Correct = Pred.eq((Targets.reshape(-1) > 0.5).float()).float()

        # Bin by Confidence, Accumulate Running Totals
        for i, (Low, High) in enumerate(zip(Edges[:-1], Edges[1:])):
            Bin = (Conf > Low) & (Conf <= High)
            Bin_Correct[i] += Correct[Bin].sum()
            Bin_Conf[i]    += Conf[Bin].sum()
            Bin_Count[i]   += Bin.sum()

## Accumulate Bin Statistics for Expected Calibration Error over Predicted-Positive Voxels
def Accumulate_Bin_Stats_Positive(Logits: torch.Tensor, Targets: torch.Tensor,
                                   Bin_Correct: torch.Tensor, Bin_Conf: torch.Tensor,
                                   Bin_Count: torch.Tensor, Edges: torch.Tensor) -> None:

    with torch.no_grad():

        Prob   = torch.sigmoid(Logits).reshape(-1)
        Actual = (Targets.reshape(-1) > 0.5).float()

        # Select on PREDICTIONS
        Select = Prob > 0.5
        if Select.sum() == 0:
            return
        Prob, Actual = Prob[Select], Actual[Select]

        # Bin by Predicted Probability, Accumulate Running Totals
        for i, (Low, High) in enumerate(zip(Edges[:-1], Edges[1:])):
            Bin = (Prob > Low) & (Prob <= High)
            Bin_Correct[i] += Actual[Bin].sum()
            Bin_Conf[i]    += Prob[Bin].sum()
            Bin_Count[i]   += Bin.sum()

## Compute Weighted ECE from Accumulated Bin Statistics
def Weighted_ECE(Bin_Correct: torch.Tensor, Bin_Conf: torch.Tensor, Bin_Count: torch.Tensor) -> float:

    Total = Bin_Count.sum().clamp(min=1)
    Error = ((Bin_Correct / Bin_Count.clamp(min=1)) - (Bin_Conf / Bin_Count.clamp(min=1))).abs()
    return (Error * (Bin_Count / Total)).nan_to_num(0).sum().item()

## Compute NLL, ECE, and ECE_Positive over the Dataset
def Streaming_Metrics(Model: torch.nn.Module, Dataset: torch.utils.data.DataLoader,
                       Device: torch.device, Log_T: torch.nn.Parameter,
                       Criterion: torch.nn.Module, Apply_T: bool,
                       Problem_Type: str, Previous_State_Idx: int, N_Bins: int = 20) -> tuple:

    NLL_Sum, NLL_N = 0.0, 0
    Bin_Correct,     Bin_Conf,     Bin_Count     = torch.zeros(N_Bins, device=Device), torch.zeros(N_Bins, device=Device), torch.zeros(N_Bins, device=Device)
    Bin_Correct_Pos, Bin_Conf_Pos, Bin_Count_Pos = torch.zeros(N_Bins, device=Device), torch.zeros(N_Bins, device=Device), torch.zeros(N_Bins, device=Device)
    Edges = torch.linspace(0.5, 1.0, N_Bins + 1, device=Device)

    with torch.no_grad():

        # Loop over Batches
        for Input_Batch, Output_Batch, _, _, _, _, _, _, _ in Dataset:

            # Move to Device
            Input_Batch  = Input_Batch.to(Device, dtype=torch.float32, non_blocking=True)
            Output_Batch = Output_Batch.to(Device, dtype=torch.float32, non_blocking=True)

            # Forward Pass
            Logits = Model(Input_Batch, apply_temperature=False)
            if Apply_T:
                Logits = Logits / Log_T.exp()

            Target = _Get_Target(Output_Batch, Input_Batch, Problem_Type, Previous_State_Idx)

            # Accumulate NLL Weighted by Voxel Count
            NLL_Sum += Criterion(Logits, Target).item() * Logits.numel()
            NLL_N   += Logits.numel()

            # Accumulate Calibration Bin Statistics
            Accumulate_Bin_Stats(Logits, Target, Bin_Correct, Bin_Conf, Bin_Count, Edges)
            Accumulate_Bin_Stats_Positive(Logits, Target, Bin_Correct_Pos, Bin_Conf_Pos, Bin_Count_Pos, Edges)

            # Release Per-Batch Tensors
            del Input_Batch, Output_Batch, Logits, Target

        torch.cuda.empty_cache()

    # Combine Bin Statistics into Weighted ECE
    ECE     = Weighted_ECE(Bin_Correct, Bin_Conf, Bin_Count)
    ECE_Pos = Weighted_ECE(Bin_Correct_Pos, Bin_Conf_Pos, Bin_Count_Pos)

    return NLL_Sum / max(NLL_N, 1), ECE, ECE_Pos

## Define Temperature Scaling Function
def TemperatureScaling(Model: torch.nn.Module, Dataset: torch.utils.data.DataLoader,
                       Device: torch.device, Problem_Type: str, Input_Properties: list,
                       N_Epochs: int = 3, LR: float = 0.01) -> torch.nn.Module:
    """
    Calibrates Model.log_temperature by fitting a scalar temperature against
    the model actually outputs -- Growth logits for Incremental
    problems, full-state logits for Static. 
    """

    Previous_State_Idx = (Input_Properties.index('previous_consumed_cavity')
                          if Problem_Type == "Incremental" else None)

    Log_T = Model.log_temperature
    Log_T.data.zero_()
    Log_T.requires_grad_(True)

    # Freeze Model
    Was_Training = Model.training
    Model.eval()
    for Name, Param in Model.named_parameters():
        if Name not in ('log_temperature'):
            Param.requires_grad_(False)

    # Class-Imbalance Correction
    N_Pos = torch.zeros(1, device=Device)
    N_Neg = torch.zeros(1, device=Device)
    with torch.no_grad():

        # Loop over Batches
        for Input_Batch, Output_Batch, _, _, _, _, _, _, _ in Dataset:
            Input_Batch  = Input_Batch.to(Device, dtype=torch.float32, non_blocking=True)
            Output_Batch = Output_Batch.to(Device, dtype=torch.float32, non_blocking=True)
            Target = _Get_Target(Output_Batch, Input_Batch, Problem_Type, Previous_State_Idx)
            N_Pos += (Target > 0.5).sum()
            N_Neg += (Target <= 0.5).sum()
            del Input_Batch, Output_Batch, Target

        torch.cuda.empty_cache()

    Pos_Weight = (N_Neg / N_Pos.clamp(min=1)).detach()

    ## FIT CALIBRATION PARAMETER
    Criterion          = torch.nn.BCEWithLogitsLoss(pos_weight=Pos_Weight)  # Used to Fit T
    Criterion_Unweighted = torch.nn.BCEWithLogitsLoss()                     # Used for Reporting Only
    Temp_Optimizer     = torch.optim.Adam([Log_T], lr=LR)

    # Metrics Before Calibration
    NLL_Before, ECE_Before, ECE_Pos_Before = Streaming_Metrics(
        Model, Dataset, Device, Log_T, Criterion_Unweighted, Apply_T=False,
        Problem_Type=Problem_Type, Previous_State_Idx=Previous_State_Idx)

    # Mini-Batch Gradient Steps
    for Epoch in range(N_Epochs):
        for Input_Batch, Output_Batch, _, _, _, _, _, _, _ in Dataset:

            # Move to Device
            Input_Batch  = Input_Batch.to(Device, dtype=torch.float32, non_blocking=True)
            Output_Batch = Output_Batch.to(Device, dtype=torch.float32, non_blocking=True)

            # Forward Pass
            with torch.no_grad():
                Logits = Model(Input_Batch, apply_temperature=False)

            Target = _Get_Target(Output_Batch, Input_Batch, Problem_Type, Previous_State_Idx)

            # Optimization Step
            Temp_Optimizer.zero_grad()
            Loss = Criterion(Logits / Log_T.exp(), Target)
            Loss.backward()
            Temp_Optimizer.step()

            # Release Per-Batch Tensors
            del Input_Batch, Output_Batch, Logits, Target

        torch.cuda.empty_cache()

    # Lock Parameters and Activate Scaling at Inference
    Log_T.requires_grad_(False)
    Model.Temp_Toggle = True

    # Metrics After Calibration
    NLL_After, ECE_After, ECE_Pos_After = Streaming_Metrics(
        Model, Dataset, Device, Log_T, Criterion_Unweighted, Apply_T=True,
        Problem_Type=Problem_Type, Previous_State_Idx=Previous_State_Idx)

    if Was_Training:
        Model.train()

    ## Print Calibration Metrics
    print(f'\nTemperature Scaling: T = {Log_T.exp().item():.4f}  (Pos_Weight = {Pos_Weight.item():.2f})')
    print(f"{'NLL':<20} Before: {NLL_Before:.4f}  After: {NLL_After:.4f}")
    print(f"{'ECE':<20} Before: {ECE_Before:.4f}  After: {ECE_After:.4f}")
    print(f"{'ECE_Positive':<20} Before: {ECE_Pos_Before:.4f}  After: {ECE_Pos_After:.4f}")

    return Model