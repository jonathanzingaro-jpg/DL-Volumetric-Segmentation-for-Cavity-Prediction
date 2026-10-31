## Functions/Training/CustomMetrics.py

## Import Libraries
import torch
import torch.nn.functional as F
from typing import List


## Loss
class SegmentationLoss(torch.nn.Module):
    """
    Configurable segmentation loss. Loss_Type:
    'MSE', 'BCE', 'Dice', 'Soft_IoU', 'Soft_IoU+BCE', 'Dice+IoU', 'Tversky'.

    Returns per-sample loss, shape (B,).

    Comments: 
    - Pred is always raw logits the model's plain forward() output
    (Growth logits, in training mode, for Incremental problems).
    - Non-Incremental Problems (Problem_Type != "Incremental"): Target is used as-is.
    - Incremental Problems (Problem_Type == "Incremental"): Target passed in is the 
    FULL state (Target_State). Growth_Target = Target_State - Previous_State
    """

    def __init__(self, Loss_Type: str, Problem_Type: str, Input_Properties: List, Smooth: float = 1.0,
                 Alpha: float = 0.3, Beta: float = 0.7, Lambda: float = 0.5):
        super().__init__()
        self.Loss_Type = Loss_Type
        self.Problem_Type = Problem_Type
        self.Smooth = Smooth
        self.Alpha = Alpha
        self.Beta = Beta
        self.Lambda = Lambda

        # Only required when running an Incremental problem.
        if self.Problem_Type == "Incremental":
            self.Previous_State_Idx = Input_Properties.index('previous_consumed_cavity')
        else:
            self.Previous_State_Idx = None

    def forward(self, Pred: torch.Tensor, Target: torch.Tensor, Input: torch.Tensor = None) -> torch.Tensor:

        # Incremental
        if self.Problem_Type == "Incremental":
            if Input is None:
                raise ValueError("SegmentationLoss is Incremental but no Input tensor was provided to forward().")
            Previous_State = Input[..., self.Previous_State_Idx].unsqueeze(-1)
            Target = (Target - Previous_State).clamp(min=0.0, max=1.0)

        Raw_Pred_Flat = Pred.reshape(Pred.shape[0], -1)
        Raw_Target_Flat = Target.reshape(Target.shape[0], -1)
        Prob_Flat = torch.sigmoid(Raw_Pred_Flat)

        if self.Loss_Type == 'MSE':
            return F.mse_loss(Prob_Flat, Raw_Target_Flat, reduction='none').mean(dim=1)

        if self.Loss_Type == 'BCE':
            return F.binary_cross_entropy_with_logits(Prob_Flat, Raw_Target_Flat, reduction='none').mean(dim=1)

        Intersection = (Prob_Flat * Raw_Target_Flat).sum(dim=1)
        Pred_Sum = Prob_Flat.sum(dim=1)
        Target_Sum = Raw_Target_Flat.sum(dim=1)

        if self.Loss_Type == 'Dice':
            return 1 - (2 * Intersection + self.Smooth) / (Pred_Sum + Target_Sum + self.Smooth)

        if self.Loss_Type == 'Soft_IoU':
            return 1 - (Intersection + self.Smooth) / (Pred_Sum + Target_Sum - Intersection + self.Smooth)

        if self.Loss_Type == 'Dice+IoU':
            Dice_Loss = 1 - (2 * Intersection + self.Smooth) / (Pred_Sum + Target_Sum + self.Smooth)
            IoU_Loss = 1 - (Intersection + self.Smooth) / (Pred_Sum + Target_Sum - Intersection + self.Smooth)
            return (Dice_Loss + IoU_Loss) ** 2

        if self.Loss_Type == 'Soft_IoU+BCE':
            BCE = F.binary_cross_entropy_with_logits(Prob_Flat, Raw_Target_Flat, reduction='none').mean(dim=1)
            IoU_Loss = (1 - (Intersection + self.Smooth) / (Pred_Sum + Target_Sum - Intersection + self.Smooth)) ** 2
            return IoU_Loss + self.Lambda * BCE

        if self.Loss_Type == 'Tversky':
            # Alpha penalises false positives, Beta false negatives.
            False_Positive = Pred_Sum - Intersection
            False_Negative = Target_Sum - Intersection
            Tversky = (Intersection + self.Smooth) / (
                Intersection + self.Alpha * False_Positive + self.Beta * False_Negative + self.Smooth)
            return 1 - Tversky

        raise ValueError(f"Unknown Loss_Type: '{self.Loss_Type}'")


## Metrics
class SegmentationMetrics(torch.nn.Module):
    """
    IoU, FScore, Accuracy and tonnage error, computed from Pred, Target,
    Density and Uranium. Returns per-sample tensors, shape (B,).
    """

    def __init__(self, Problem_Type: str, Input_Properties: List, Threshold: float = 0.5,
                 Smooth: float = 1.0, Growth_Smooth: float = 1e-3, Mass_Floor: float = 1.0):
        super().__init__()
        self.Problem_Type = Problem_Type
        self.Threshold = Threshold
        self.Smooth = Smooth
        self.Growth_Smooth = Growth_Smooth
        self.Mass_Floor = Mass_Floor

        # Previous State Index -> Only Valid for Incremental Problems
        if self.Problem_Type == "Incremental":
            self.Previous_State_Idx = Input_Properties.index('previous_consumed_cavity')
        else:
            self.Previous_State_Idx = None

    @staticmethod
    def _Binary_Metrics(Pred_Binary: torch.Tensor, Target_Binary: torch.Tensor, Smooth: float) -> dict:
        True_Positive = (Pred_Binary * Target_Binary).sum(dim=1)
        Pred_Sum = Pred_Binary.sum(dim=1)
        Target_Sum = Target_Binary.sum(dim=1)

        IoU = (True_Positive + Smooth) / (Pred_Sum + Target_Sum - True_Positive + Smooth)
        FScore = (2 * True_Positive + Smooth) / (Pred_Sum + Target_Sum + Smooth)
        Accuracy = (Pred_Binary == Target_Binary).float().mean(dim=1)

        return {'IoU': IoU, 'FScore': FScore, 'Accuracy': Accuracy}

    @torch.no_grad()
    def forward(self, Pred: torch.Tensor, Target: torch.Tensor,
                Density: torch.Tensor, Uranium: torch.Tensor,
                Input: torch.Tensor = None) -> dict:

        # Non-Incremental: Standard Metrics
        if self.Problem_Type != "Incremental":
            Pred_Binary = (Pred.reshape(Pred.shape[0], -1) > self.Threshold).float()
            Target_Binary = (Target.reshape(Target.shape[0], -1) > self.Threshold).float()

            Standard_Metrics = self._Binary_Metrics(Pred_Binary, Target_Binary, self.Smooth)
            IoU = Standard_Metrics['IoU']
            FScore = Standard_Metrics['FScore']
            Accuracy = Standard_Metrics['Accuracy']
            GrowthIoU = None

        # Incremental
        else:
            if Input is None:
                raise ValueError("SegmentationMetrics is Incremental but no Input tensor was provided to forward().")

            Previous_State = Input[..., self.Previous_State_Idx].unsqueeze(-1)

            # GrowthIoU
            Growth_Target = (Target - Previous_State).clamp(min=0.0, max=1.0)
            Growth_Pred_Binary = (Pred.reshape(Pred.shape[0], -1) > self.Threshold).float()
            Growth_Target_Binary = (Growth_Target.reshape(Growth_Target.shape[0], -1) > self.Threshold).float()

            GrowthIoU = self._Binary_Metrics(
                Growth_Pred_Binary, Growth_Target_Binary, self.Growth_Smooth)['IoU']

            # Full-State IoU
            Full_State_Prob = (Previous_State + Pred).clamp(0.0, 1.0)
            Full_State_Pred_Binary = (Full_State_Prob.reshape(Full_State_Prob.shape[0], -1) > self.Threshold).float()
            Full_State_Target_Binary = (Target.reshape(Target.shape[0], -1) > self.Threshold).float()

            Full_State_Metrics = self._Binary_Metrics(
                Full_State_Pred_Binary, Full_State_Target_Binary, self.Smooth)
            IoU = Full_State_Metrics['IoU']
            FScore = Full_State_Metrics['FScore']
            Accuracy = Full_State_Metrics['Accuracy']

            Pred_Binary = Full_State_Pred_Binary
            Target_Binary = Full_State_Target_Binary

        Grade = Density.reshape(Density.shape[0], -1) * Uranium.reshape(Uranium.shape[0], -1)
        Pred_Mass = (Pred_Binary * Grade).sum(dim=1)
        Target_Mass = (Target_Binary * Grade).sum(dim=1)
        Tonnage_Error = (Pred_Mass - Target_Mass) / Target_Mass.clamp(min=self.Mass_Floor)

        # Build Results Dict 
        Results = {'IoU': IoU}
        if GrowthIoU is not None:
            Results['GrowthIoU'] = GrowthIoU
        Results.update({
            'FScore':            FScore,
            'Accuracy':          Accuracy,
            'Tonnage_Error':     Tonnage_Error,
            'Tonnage_Error_Abs': Tonnage_Error.abs(),
            'Pred_Mass':         Pred_Mass,
            'Target_Mass':       Target_Mass,
        })

        return Results