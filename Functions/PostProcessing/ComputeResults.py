# Functions/PostProcessing/ComputeResults.py

# Import Libraries
import math
import torch
import numpy
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

# Project Functions
from Functions.Preprocessing.DataNormalization import *
from Functions.PostProcessing.Plotting import PlotPrediction, PlotStatisticsCurve

# Number of threshold bins torchmetrics uses internally 
AUC_THRESHOLD_BINS = 200


def _Confusion_Counts(Preds_Binary: numpy.ndarray, Gt_Binary: numpy.ndarray) -> tuple:
    """Voxel-level TN/FP/FN/TP."""
    Tp = int(numpy.count_nonzero(Preds_Binary & Gt_Binary))
    Fp = int(numpy.count_nonzero(Preds_Binary & ~Gt_Binary))
    Fn = int(numpy.count_nonzero(~Preds_Binary & Gt_Binary))
    Tn = int(Gt_Binary.size - Tp - Fp - Fn)
    return Tn, Fp, Fn, Tp


def _Compute_Metrics_Block(IoU_Arr: numpy.ndarray, FScore_Arr: numpy.ndarray,
                           Accuracy_Arr: numpy.ndarray, Loss_Arr: numpy.ndarray,
                           Tonnage_Arr: numpy.ndarray, Tp: int, Tn: int, Fp: int, Fn: int,
                           Base_Rate: float, Roc_Auc: float, Pr_Auc: float,
                           GrowthIoU_Arr: numpy.ndarray = None) -> dict:
    """
    Provides a dict with all the results for the split passed into the function.
    """
    if len(IoU_Arr) == 0:
        return None

    N = float(Tp + Tn + Fp + Fn)
    Mcc_Denominator = math.sqrt(float(Tp + Fp) * float(Tp + Fn) * float(Tn + Fp) * float(Tn + Fn))
    Mcc = float((Tp * Tn - Fp * Fn) / Mcc_Denominator) if Mcc_Denominator > 0 else float('nan')

    P_Observed = (Tp + Tn) / N if N else float('nan')
    P_Expected = ((Tp + Fp) * (Tp + Fn) + (Tn + Fn) * (Tn + Fp)) / (N * N) if N else float('nan')
    Cohen_Kappa = float((P_Observed - P_Expected) / (1.0 - P_Expected)) if N and P_Expected < 1.0 else float('nan')

    Results = {
        "Loss": float(numpy.mean(Loss_Arr)),
        "Loss_Std": float(numpy.std(Loss_Arr)),
        "IoU": float(numpy.mean(IoU_Arr)),
        "IoU_Std": float(numpy.std(IoU_Arr)),
        "IoU_Variance": float(numpy.var(IoU_Arr)),
        "IoU_Min": float(numpy.min(IoU_Arr)),
        "IoU_Max": float(numpy.max(IoU_Arr)),
        "FScore": float(numpy.mean(FScore_Arr)),
        "FScore_Std": float(numpy.std(FScore_Arr)),
        "Accuracy": float(numpy.mean(Accuracy_Arr)),
        "Accuracy_Std": float(numpy.std(Accuracy_Arr)),
        "Tonnage_Error": float(numpy.mean(Tonnage_Arr)),
        "Tonnage_Error_Std": float(numpy.std(Tonnage_Arr)),
        "Tonnage_Error_Abs": float(numpy.mean(numpy.abs(Tonnage_Arr))),
        "N_Samples": int(len(IoU_Arr)),
        "TP": int(Tp), "TN": int(Tn), "FP": int(Fp), "FN": int(Fn),
        "Precision": float(Tp / (Tp + Fp) if (Tp + Fp) > 0 else 0.0),
        "Recall": float(Tp / (Tp + Fn) if (Tp + Fn) > 0 else 0.0),
        "Specificity": float(Tn / (Tn + Fp) if (Tn + Fp) > 0 else 0.0),
        "Balanced_Accuracy": float(((Tp / (Tp + Fn)) + (Tn / (Tn + Fp))) / 2
                                   if (Tp + Fn) > 0 and (Tn + Fp) > 0 else 0.0),
        "MCC": Mcc,
        "Cohen_Kappa": Cohen_Kappa,
        "Base_Rate": float(Base_Rate),
        "ROC_AUC": Roc_Auc,
        "PR_AUC": Pr_Auc,
    }

    if GrowthIoU_Arr is not None and len(GrowthIoU_Arr) > 0:
        Results["GrowthIoU"] = float(numpy.mean(GrowthIoU_Arr))
        Results["GrowthIoU_Std"] = float(numpy.std(GrowthIoU_Arr))

    return Results


def _Print_Results_Block(Results: dict, Title: str) -> None:
    print("\n" + "=" * 80)
    print(Title)
    print("=" * 80)
    if Results is None:
        print("  (No samples in this dataset)")
        print("=" * 80 + "\n")
        return

    print(f"\n  --- Basic Metrics ({Results['N_Samples']} samples) ---")
    print(f"  Loss:             {Results['Loss']:.6f} ± {Results['Loss_Std']:.6f}")
    print(f"  IoU:              {Results['IoU']:.6f} ± {Results['IoU_Std']:.6f}")
    print(f"  IoU Range:        [{Results['IoU_Min']:.6f}, {Results['IoU_Max']:.6f}]")
    if 'GrowthIoU' in Results:
        print(f"  GrowthIoU:        {Results['GrowthIoU']:.6f} ± {Results['GrowthIoU_Std']:.6f}")
    print(f"  F-Score:          {Results['FScore']:.6f} ± {Results['FScore_Std']:.6f}")
    print(f"  Accuracy:         {Results['Accuracy']:.6f} ± {Results['Accuracy_Std']:.6f}")

    print(f"\n  --- Tonnage ---")
    print(f"  Signed:           {Results['Tonnage_Error']:.6f} ± {Results['Tonnage_Error_Std']:.6f}")
    print(f"  Absolute:         {Results['Tonnage_Error_Abs']:.6f}")

    print(f"\n  --- Confusion Matrix ---")
    print(f"  True Positives:   {Results['TP']:,}")
    print(f"  True Negatives:   {Results['TN']:,}")
    print(f"  False Positives:  {Results['FP']:,}")
    print(f"  False Negatives:  {Results['FN']:,}")

    print(f"\n  --- Ranking Metrics ---")
    print(f"  ROC AUC:          {Results['ROC_AUC']:.6f}")
    print(f"  PR AUC (AP):      {Results['PR_AUC']:.6f}")
    print(f"  Base rate:        {Results.get('Base_Rate', float('nan')):.6g}")
    print(f"  Precision:        {Results['Precision']:.6f}")
    print(f"  Recall:           {Results['Recall']:.6f}")
    print(f"  Specificity:      {Results['Specificity']:.6f}")
    print(f"  Balanced Acc:     {Results['Balanced_Accuracy']:.6f}")
    print(f"  MCC:              {Results['MCC']:.6f}")
    print(f"  Cohen's Kappa:    {Results['Cohen_Kappa']:.6f}")
    print("=" * 80 + "\n")


def ComputeResults(Loss, Metrics, Device: torch.device, Dataset: torch.utils.data.DataLoader,
                    Model_List: list, Result_Path: str,
                    Plot_Test_Case: bool, Plot_Performance: bool,
                    Output_Keys: str, Split_Name: str):
    """
    Runs an ensemble over Dataset, averaging predictions across members, then
    reports overall loss and metrics.
    """

    Num_Ensembles = len(Model_List)

    Model_List = [Model.to(device=Device, dtype=torch.float32) for Model in Model_List]
    for Model in Model_List:
        Model.eval()

    Criterion = Loss
    Metrics_Fn = Metrics
    Roc_Auc_Metric = BinaryAUROC(thresholds=AUC_THRESHOLD_BINS).to(Device)
    Pr_Auc_Metric  = BinaryAveragePrecision(thresholds=AUC_THRESHOLD_BINS).to(Device)

    Dataset_Name_All = []
    Backfill_All = []
    Loss_All = []
    IoU_All, FScore_All, Accuracy_All = [], [], []
    GrowthIoU_All = None 
    Tonnage_Error_All = []
    Pred_Prob_All, Gt_Binary_All = [], []  # Raw Voxel Scores for PlotStatisticsCurve

    # Running Confusion Counts
    Counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'Pos': 0, 'Total': 0}

    Num_Batches = 0

    with torch.inference_mode():
        for (Input_Batch, Output_Batch, Dataset_Name_Batch, Coord_Batch,
             Backfill_Batch, _, _, Density_Batch, Uranium_Batch) in Dataset:

            Input_Batch    = Input_Batch.to(Device, dtype=torch.float32, non_blocking=True)
            Output_Batch   = Output_Batch.to(Device, dtype=torch.float32, non_blocking=True)
            Backfill_Batch = Backfill_Batch.to(Device, dtype=torch.float32, non_blocking=True)
            Density_Batch  = Density_Batch.to(Device, dtype=torch.float32, non_blocking=True)
            Uranium_Batch  = Uranium_Batch.to(Device, dtype=torch.float32, non_blocking=True)

            Output_Batch_Pred = torch.zeros_like(Output_Batch)

            for Model in Model_List:
                Logits = Model(Input_Batch)
                Output_Batch_Pred += torch.sigmoid(Logits)

            Output_Batch_Pred = Output_Batch_Pred / Num_Ensembles
            Output_Batch_Pred_Logits = torch.logit(Output_Batch_Pred.clamp(1e-6, 1 - 1e-6))

            # Scoring_Pred / Scoring_Target: Model Score for Aggragated Metrics
            # Display_Pred / Display_Target: Voxel Field for Visualization
            if Metrics_Fn.Problem_Type == "Incremental":
                Previous_State = Input_Batch[..., Metrics_Fn.Previous_State_Idx].unsqueeze(-1)
                Scoring_Pred   = Output_Batch_Pred
                Scoring_Target = (Output_Batch - Previous_State).clamp(0.0, 1.0)
                Display_Pred   = (Previous_State + Output_Batch_Pred).clamp(0.0, 1.0)
                Display_Target = Output_Batch
            else:
                # Static: Output_Batch_Pred -> Full State
                Scoring_Pred   = Output_Batch_Pred
                Scoring_Target = Output_Batch
                Display_Pred   = Output_Batch_Pred
                Display_Target = Output_Batch

            # Voxel-Level Tensors for Batch 
            Pred_Prob_Flat  = Scoring_Pred.reshape(-1)
            Gt_Binary_Flat  = (Scoring_Target.reshape(-1) > Metrics_Fn.Threshold)

            # Running Confusion Counts
            Pred_Prob_Flat_Np = Pred_Prob_Flat.cpu().numpy()
            Gt_Flat_Np        = Gt_Binary_Flat.cpu().numpy()
            Pred_Bin_Flat_Np  = Pred_Prob_Flat_Np > Metrics_Fn.Threshold
            Tn_B, Fp_B, Fn_B, Tp_B = _Confusion_Counts(Pred_Bin_Flat_Np, Gt_Flat_Np)
            Counts['TP'] += Tp_B; Counts['TN'] += Tn_B
            Counts['FP'] += Fp_B; Counts['FN'] += Fn_B
            Counts['Pos'] += int(Gt_Flat_Np.sum())
            Counts['Total'] += Gt_Flat_Np.size

            if Plot_Performance is True:
                Pred_Prob_All.append(Pred_Prob_Flat_Np)
                Gt_Binary_All.append(Gt_Flat_Np)

            # ROC/PR AUC Accumulators
            Roc_Auc_Metric.update(Pred_Prob_Flat, Gt_Binary_Flat)
            Pr_Auc_Metric.update(Pred_Prob_Flat, Gt_Binary_Flat)

            # Metrics (per-sample)
            Metrics_Batch = Metrics_Fn(Output_Batch_Pred, Output_Batch, Density_Batch, Uranium_Batch, Input_Batch)

            # Loss
            Loss_Batch = Criterion(Output_Batch_Pred_Logits, Output_Batch, Input_Batch)

            Num_Batches += 1
            B = Output_Batch.shape[0]

            Dataset_Name_All.extend(Dataset_Name_Batch)
            Backfill_All.extend((Backfill_Batch.flatten(1).sum(dim=1) > 0).cpu().numpy())
            Loss_All.extend(Loss_Batch.reshape(-1).cpu().numpy())
            IoU_All.extend(Metrics_Batch['IoU'].reshape(-1).cpu().numpy())
            FScore_All.extend(Metrics_Batch['FScore'].reshape(-1).cpu().numpy())
            Accuracy_All.extend(Metrics_Batch['Accuracy'].reshape(-1).cpu().numpy())
            Tonnage_Error_All.extend(Metrics_Batch['Tonnage_Error'].reshape(-1).cpu().numpy())

            # GrowthIoU
            if 'GrowthIoU' in Metrics_Batch:
                if GrowthIoU_All is None:
                    GrowthIoU_All = []
                GrowthIoU_All.extend(Metrics_Batch['GrowthIoU'].reshape(-1).cpu().numpy())

            if Plot_Test_Case is True:
                Results_Samples = {"IoU": Metrics_Batch['IoU'].reshape(-1).cpu().numpy().tolist(),
                                   "FScore": Metrics_Batch['FScore'].reshape(-1).cpu().numpy().tolist(),
                                   "Accuracy": Metrics_Batch['Accuracy'].reshape(-1).cpu().numpy().tolist(),
                                   "Loss": Loss_All[-B:]}
                if 'GrowthIoU' in Metrics_Batch:
                    Results_Samples["GrowthIoU"] = Metrics_Batch['GrowthIoU'].reshape(-1).cpu().numpy().tolist()
                PlotPrediction(Display_Pred, Display_Target, Dataset_Name_Batch, Coord_Batch,
                               Backfill_Batch, Results_Samples, Result_Path,
                               Ensemble_Toggle=True, Output_Keys=Output_Keys)

    if Num_Batches == 0:
        return {"Loss": float('nan'), "IoU": float('nan'),
                "FScore": float('nan'), "Accuracy": float('nan')}

    Loss_All     = numpy.array(Loss_All)
    IoU_All      = numpy.array(IoU_All)
    FScore_All   = numpy.array(FScore_All)
    Accuracy_All = numpy.array(Accuracy_All)
    Tonnage_Error_All = numpy.array(Tonnage_Error_All)
    GrowthIoU_All_Arr = numpy.array(GrowthIoU_All) if GrowthIoU_All is not None else None

    Roc_Auc_All = float(Roc_Auc_Metric.compute())
    Pr_Auc_All  = float(Pr_Auc_Metric.compute())

    Results = _Compute_Metrics_Block(
        IoU_All, FScore_All, Accuracy_All, Loss_All, Tonnage_Error_All,
        Counts['TP'], Counts['TN'], Counts['FP'], Counts['FN'],
        Base_Rate=(Counts['Pos'] / Counts['Total']) if Counts['Total'] else float('nan'),
        Roc_Auc=Roc_Auc_All, Pr_Auc=Pr_Auc_All, GrowthIoU_Arr=GrowthIoU_All_Arr)

    if Plot_Performance is True:
        Pred_Prob_All = numpy.concatenate(Pred_Prob_All)
        Gt_Binary_All = numpy.concatenate(Gt_Binary_All)
        PlotStatisticsCurve(Pred_Prob_All, Gt_Binary_All, Result_Path)

    _Print_Results_Block(Results, f"ENSEMBLE EVALUATION RESULTS {Split_Name} -- OVERALL")

    return dict(Results)