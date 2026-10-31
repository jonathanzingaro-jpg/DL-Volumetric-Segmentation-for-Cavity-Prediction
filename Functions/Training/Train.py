## Functions/Training/Train.py

## Import Libraries 
import os
import torch
from tqdm import tqdm
from scipy.io import savemat

# Import Custom Functions
from Functions.Training.RunEpoch import RunEpoch
from Functions.Training.TemperatureScaling import TemperatureScaling
from Functions.PostProcessing.Plotting import PostProcessPlotting
from Functions.PostProcessing.SHAPAnalysis import ShapAnalysis

## Metric Key Lookup
def _Get(Metrics: dict, Name: str, Default=float('nan')):
    for Candidate in (Name, Name.replace('_', ' '), Name.replace(' ', '_')):
        if Candidate in Metrics:
            return float(Metrics[Candidate])
    return Default

## Build TQDM Postfix 
_POSTFIX_METRICS = ('Loss', 'IoU', 'GrowthIoU', 'FScore', 'Accuracy', 'Tonnage_Error_Abs')

def _Build_Postfix(Train_Metrics: dict, Valid_Metrics: dict, Test_Metrics: dict) -> str:
    Parts = []
    for Key in _POSTFIX_METRICS:
        if Key not in Train_Metrics:
            continue  # e.g. GrowthIoU absent for non-Incremental runs
        Parts.append(f"{Key}[{_Get(Train_Metrics, Key):.3f}, {_Get(Valid_Metrics, Key):.3f}, {_Get(Test_Metrics, Key):.3f}]")
    return ", ".join(Parts)

## Define Training Function
def Train(Paths, Model, Optimizer, Scheduler, Datasets,
          Model_Config, Toggle_Config, Device, Params, Input_Properties, Loss, Metrics, Data_Config):

    # Extract Variables from Inputs
    Model_Path, Result_Path = Paths['Model_Path'], Paths['Result_Path']
    Train_Dataset  = Datasets['Train']
    Valid_Dataset  = Datasets['Valid']
    Test_Dataset   = Datasets['Test']
    Binary_Channel_Idx = Datasets.get('Binary_Channel_Idx', [])

    Model_Name   = Model_Config['Model_Name']
    N_Epoch      = Model_Config['N_Epoch']
    Loss_Type    = Model_Config['Loss_Type']
    Temp_Scaling = Model_Config['Temp_Scaling']
    Task_Type    = Model_Config['Task_Type']
    Problem_Type = Model_Config['Problem_Type']
    SHAP_Toggle = Toggle_Config['SHAP_Toggle']

    Augmentation_Toggle = Data_Config['Augmentation_Toggle']
    Noise_Toggle        = Data_Config['Noise_Toggle']
    Noise_Lvl           = Data_Config['Noise_Level']

    Is_Incremental = (Problem_Type == "Incremental")

    # Define Empty Loss Tensors [Training, Validation, Test]
    Epoch_Tensor    = torch.zeros(N_Epoch, 1)
    Loss_Tensor     = torch.zeros(N_Epoch, 3)
    IoU_Tensor      = torch.zeros(N_Epoch, 3)
    FScore_Tensor   = torch.zeros(N_Epoch, 3)
    Accuracy_Tensor = torch.zeros(N_Epoch, 3)
    Tonnage_Error_Tensor = torch.zeros(N_Epoch, 3)
    Growth_IoU_Tensor = torch.zeros(N_Epoch, 3) if Is_Incremental else None         # Only used for Incremental Problems

    # Move Model to Device
    Model = Model.to(Device, dtype=torch.float32)

    # Define Progress Bar
    pbar = tqdm(range(N_Epoch), desc="Progress", unit="epoch",
                dynamic_ncols=True, leave=True)

    # Declare Best Score Variable and Best-Epoch Cache
    Max_IoU      = -float("inf")
    Best_Epoch   = 0
    Best_Metrics = {Split: {} for Split in ('Train', 'Valid', 'Test')}

    ## TRAINING LOOP
    for Epoch in pbar:

        ## RUN THROUGH TRAINING SET
        Train_Metrics = RunEpoch(
            Model, Optimizer, Train_Dataset, Loss, Metrics, Device, Train_Toggle=True,
            Binary_Channel_Idx=Binary_Channel_Idx,
            Augmentation_Toggle=Augmentation_Toggle,
            Noise_Toggle=Noise_Toggle,
            Noise_Lvl=Noise_Lvl,
        )

        ## RUN THROUGH VALIDATION SET 
        Valid_Metrics = RunEpoch(Model, Optimizer, Valid_Dataset, Loss, Metrics, Device, Train_Toggle=False)

        ## RUN THROUGH TEST SET
        Test_Metrics = RunEpoch(Model, Optimizer, Test_Dataset, Loss, Metrics, Device, Train_Toggle=False)

        # Save Metrics over Epochs
        Epoch_Tensor[Epoch]       = Epoch
        Loss_Tensor[Epoch, :]     = torch.tensor([_Get(Train_Metrics, 'Loss'), _Get(Valid_Metrics, 'Loss'), _Get(Test_Metrics, 'Loss')])
        IoU_Tensor[Epoch, :]      = torch.tensor([_Get(Train_Metrics, 'IoU'), _Get(Valid_Metrics, 'IoU'), _Get(Test_Metrics, 'IoU')])
        FScore_Tensor[Epoch, :]   = torch.tensor([_Get(Train_Metrics, 'FScore'), _Get(Valid_Metrics, 'FScore'), _Get(Test_Metrics, 'FScore')])
        Accuracy_Tensor[Epoch, :] = torch.tensor([_Get(Train_Metrics, 'Accuracy'), _Get(Valid_Metrics, 'Accuracy'), _Get(Test_Metrics, 'Accuracy')])
        Tonnage_Error_Tensor[Epoch, :] = torch.tensor([_Get(Train_Metrics, 'Tonnage_Error'), _Get(Valid_Metrics, 'Tonnage_Error'), _Get(Test_Metrics, 'Tonnage_Error')])

        if Is_Incremental:
            Growth_IoU_Tensor[Epoch, :] = torch.tensor([_Get(Train_Metrics, 'GrowthIoU'), _Get(Valid_Metrics, 'GrowthIoU'), _Get(Test_Metrics, 'GrowthIoU')])

        ## Save Best Model - Highest Validation IoU
        Valid_IoU = _Get(Valid_Metrics, 'IoU')
        if Valid_IoU > Max_IoU:
            Max_IoU = Valid_IoU

            if isinstance(Model, torch.nn.DataParallel):
                torch.save(Model.module.state_dict(), os.path.join(Model_Path, f'{Model_Name}.pt'))
            else:
                torch.save(Model.state_dict(), os.path.join(Model_Path, f'{Model_Name}.pt'))

            # Cache Best-Epoch Metrics
            Best_Epoch = Epoch
            Best_Metrics = {
                'Train': {Key: float(Value) for Key, Value in Train_Metrics.items() if not isinstance(Value, dict)},
                'Valid': {Key: float(Value) for Key, Value in Valid_Metrics.items() if not isinstance(Value, dict)},
                'Test':  {Key: float(Value) for Key, Value in Test_Metrics.items()  if not isinstance(Value, dict)},
            }

        # Update TQDM Bar
        pbar.set_postfix_str(_Build_Postfix(Train_Metrics, Valid_Metrics, Test_Metrics))

        # Learning Rate Scheduler Update
        if Scheduler is None:
            pass
        elif type(Scheduler).__name__ == 'ReduceLROnPlateau':
            Scheduler.step(_Get(Valid_Metrics, 'IoU'))
        else:
            Scheduler.step()

    ## Reload Best Model
    Checkpoint = torch.load(os.path.join(Model_Path, f'{Model_Name}.pt'), weights_only=True)
    if isinstance(Model, torch.nn.DataParallel):
        Model.module.load_state_dict(Checkpoint)
    else:
        Model.load_state_dict(Checkpoint)

    # Run Temperature Scaling
    if Temp_Scaling is True:

        print('\nPost-Processing Temperature Scaling ...')

        # Unwrap DataParallel
        Model = Model.module if isinstance(Model, torch.nn.DataParallel) else Model
        Model = TemperatureScaling(Model, Valid_Dataset, Device, Problem_Type, Input_Properties)

        # Save Model
        torch.save(Model.state_dict(), os.path.join(Model_Path, f'{Model_Name}.pt'))

        ## Re-Evaluate
        Calib_Metrics = {
            'Train': RunEpoch(Model, Optimizer, Train_Dataset, Loss, Metrics, Device, Train_Toggle=False),
            'Valid': RunEpoch(Model, Optimizer, Valid_Dataset, Loss, Metrics, Device, Train_Toggle=False),
            'Test':  RunEpoch(Model, Optimizer, Test_Dataset, Loss, Metrics, Device, Train_Toggle=False)}

        ## Print Calibrated Metrics -- GrowthIoU only listed when Incremental
        Print_Metrics = ('Loss', 'IoU') + (('GrowthIoU',) if Is_Incremental else ()) + ('FScore', 'Accuracy', 'Tonnage_Error_Abs')
        print('\nCalibrated Metrics:')
        for Metric in Print_Metrics:
            Before = [_Get(Best_Metrics[Split], Metric) for Split in ('Train', 'Valid', 'Test')]
            After  = [_Get(Calib_Metrics[Split], Metric) for Split in ('Train', 'Valid', 'Test')]
            print(f"{Metric:<20} "
                  f"Train: {Before[0]:.4f} -> {After[0]:.4f}   "
                  f"Valid: {Before[1]:.4f} -> {After[1]:.4f}   "
                  f"Test: {Before[2]:.4f} -> {After[2]:.4f}")


    # Save metrics over epoch as .mat file for post-processing
    Best_Struct = {Split: {Key.replace(' ', '_'): Value for Key, Value in Metrics.items()}
                   for Split, Metrics in Best_Metrics.items()}

    Mat_Dict = {'Epoch_Tensor':    Epoch_Tensor.cpu().numpy(),
                'Loss_Tensor':     Loss_Tensor.cpu().numpy(),
                'IoU_Tensor':      IoU_Tensor.cpu().numpy(),
                'FScore_Tensor':   FScore_Tensor.cpu().numpy(),
                'Accuracy_Tensor': Accuracy_Tensor.cpu().numpy(),
                'Tonnage_Error_Tensor': Tonnage_Error_Tensor.cpu().numpy(),
                'Best_Epoch':      Best_Epoch,
                'Best_Metrics':    Best_Struct}

    if Is_Incremental:
        Mat_Dict['Growth_IoU_Tensor'] = Growth_IoU_Tensor.cpu().numpy()

    mat_path = os.path.join(Model_Path, f'{Model_Name}_metrics.mat')
    savemat(mat_path, Mat_Dict)

    ## Print Best Metrics
    Print_Metrics = ('Loss', 'IoU') + (('GrowthIoU',) if Is_Incremental else ()) + ('FScore', 'Accuracy', 'Tonnage_Error_Abs')
    print(f'\nBest Epoch: {Best_Epoch}')
    for Metric in Print_Metrics:
        Values = [_Get(Best_Metrics[Split], Metric) for Split in ('Train', 'Valid', 'Test')]
        if not any(Value == Value for Value in Values):  
            continue
        print(f"{Metric:<20} "
              f"Train: {Values[0]:.4f}  Valid: {Values[1]:.4f}  Test: {Values[2]:.4f}")

    # Post-Processing of Loss and Performance Metrics 
    PostProcessPlotting(Epoch_Tensor, Loss_Tensor, IoU_Tensor, FScore_Tensor, Accuracy_Tensor, Loss_Type, Result_Path, Model_Name, N_Epoch)

    ## Compute Integrated Gradients for Test Dataset
    if SHAP_Toggle is True:
        print('\nPost-Processing SHAP ...')
        ShapAnalysis(Test_Dataset, Model, Result_Path, Params['Keys']['Input_Keys'], Dataset_Type='Test')
        print('Complete!')

    return Model