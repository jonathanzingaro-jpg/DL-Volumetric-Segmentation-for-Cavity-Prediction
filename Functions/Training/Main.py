## Functions/Training/Main.py

## Import Libraries
from Functions.Imports.Training import *
from Functions.Training.Train import Train
# SegmentationLoss / SegmentationMetrics import no longer needed here


## Sequential Model Training
def TrainModels(ProcessedDir, Config, Paths, Norm_Parameters, Input_Properties):
    """One process, one device, train all ensemble members back to back."""

    Model_Config     = Config["Model_Config"]
    Optimizer_Config = Config["Optimizer_Config"]
    Scheduler_Config = Config["Scheduler_Config"]
    Toggle_Config    = Config["Toggle_Config"]

    Model_Name    = Model_Config["Model_Name"]
    N_Models = Model_Config["N_Models"]

    Device  = DeclareDevice()
    Loaders = BuildLoaders(ProcessedDir, Config["Data_Config"], Input_Properties, Num_Workers=4)

    print(f"\n=== Training {N_Models} models sequentially on {Device} ===")

    Model_List = []
    Loss, Metrics = None, None
    for Model_Number in range(1, N_Models + 1):

        print(f"\n=== Starting Training for Model: {Model_Name} | "
              f"Model {Model_Number}/{N_Models} on {Device} ===", flush=True)

        # Initialize Model, Optimizer, Scheduler, Loss and Metrics
        Model, Optimizer, Scheduler, Loss, Metrics = InitializeModel(
            Model_Config, Optimizer_Config, Scheduler_Config, Device, Norm_Parameters
        )

        # Wrap in DataParallel if multiple GPUs are available and enabled
        if Toggle_Config.get("Multi_GPU", False):
            Num_GPUs = torch.cuda.device_count() if Device.type == "cuda" else 0
            if Num_GPUs > 1:
                print(f"    -> Using DataParallel across {Num_GPUs} GPUs", flush=True)
                Model = torch.nn.DataParallel(Model)

        # Print Model Summary
        if Model_Number == 1:
            Sample_Batch = next(iter(Loaders['Train']))
            Sample_Input = Sample_Batch[0] if isinstance(Sample_Batch, (list, tuple)) else Sample_Batch
            Summary_Target = Model.module if isinstance(Model, torch.nn.DataParallel) else Model
            summary(Summary_Target, input_data=Sample_Input[:1].to(Device, dtype=torch.float32))

        # Make Directory for Model and Results
        Model_Path  = os.path.join(Paths['Base_Model_Path'],  f'Model {Model_Number}')
        Result_Path = os.path.join(Paths['Base_Result_Path'], f'Model {Model_Number}')
        os.makedirs(Model_Path,  exist_ok=True)
        os.makedirs(Result_Path, exist_ok=True)

        # Train Model
        Model = Train(
            Paths         = {'Model_Path': Model_Path, 'Result_Path': Result_Path},
            Model         = Model,
            Optimizer     = Optimizer,
            Scheduler     = Scheduler,
            Datasets      = {'Train': Loaders['Train'], 'Valid': Loaders['Valid'], 'Test': Loaders['Test'],
                             'Binary_Channel_Idx': Loaders.get('Binary_Channel_Idx', [])},
            Model_Config  = Model_Config,
            Toggle_Config = Toggle_Config,
            Device        = Device,
            Params        = Norm_Parameters,
            Input_Properties  = Input_Properties,
            Loss          = Loss,
            Metrics       = Metrics,
            Data_Config   = Config["Data_Config"],
        )

        # Unwrap DataParallel before moving to CPU / returning
        if isinstance(Model, torch.nn.DataParallel):
            Model = Model.module

        Model_List.append(Model.to("cpu"))

    return Model_List, Loss, Metrics


## Load a Single Pre-Trained Ensemble Member
def LoadSingleModel(Model_Number, Config, Device, Norm_Parameters, Paths):
    """Instantiate the architecture and load its saved weights from disk."""

    Model_Config     = Config["Model_Config"]
    Optimizer_Config = Config["Optimizer_Config"]
    Scheduler_Config = Config["Scheduler_Config"]
    N_Models    = Model_Config["N_Models"]

    print(f"\n=== Loading Pre-Trained Model: {Model_Config['Model_Name']} | "
          f"Model {Model_Number}/{N_Models} ===", flush=True)

    # Build the Model
    Model, _, _, Loss, Metrics = InitializeModel(Model_Config, Optimizer_Config, Scheduler_Config,
                                                 Device, Norm_Parameters)

    # Locate Model Checkpoint
    Model_Path      = os.path.join(Paths['Base_Model_Path'], f'Model {Model_Number}')
    Checkpoint_Path = os.path.join(Model_Path, f'{Model_Config["Model_Name"]}.pt')

    if not os.path.exists(Checkpoint_Path):
        raise FileNotFoundError(f"No saved checkpoint found at {Checkpoint_Path}. "
                                f"Cannot Skip_Training without an existing model.")

    Checkpoint = torch.load(Checkpoint_Path, map_location=Device, weights_only=False)
    State_Dict = Checkpoint.get('Model_State_Dict', Checkpoint) if isinstance(Checkpoint, dict) else Checkpoint
    Model.load_state_dict(State_Dict)

    return Model.to("cpu"), Loss, Metrics


## Load All Ensemble Members Sequentially
def LoadModels(Config, Paths, Norm_Parameters, Device):
    """One process, one device — load every ensemble member's saved weights."""

    N_Models = Config["Model_Config"]["N_Models"]
    print(f"\n=== Skipping Training: Loading {N_Models} pre-trained models on {Device} ===")

    Model_List = []
    Loss, Metrics = None, None
    for Model_Number in range(1, N_Models + 1):
        Model, Loss, Metrics = LoadSingleModel(Model_Number, Config, Device, Norm_Parameters, Paths)
        Model_List.append(Model)

    return Model_List, Loss, Metrics


## Main
def Main(CurDir, cfg):

    # Check Configuration File for Errors
    try:
        Config, Warnings = ValidateConfig(cfg)
    except ConfigError as Error:
        print(Error)
        raise SystemExit(1)

    # Extract Structures from the Validated Config
    Model_Config  = Config["Model_Config"]
    Data_Config   = Config["Data_Config"]
    Toggle_Config = Config["Toggle_Config"]

    ## Load Normalization Parameters
    Norm_Parameters = torch.load(os.path.join(CurDir, 'Dataset', Data_Config['Dataset_Name'],
                                              'params.pt'), map_location='cpu', weights_only=False)
    Input_Properties = Norm_Parameters['Keys']['Input_Keys']

    ## Path to Processed Batches
    ProcessedDir = os.path.join(CurDir, 'Dataset', Data_Config['Dataset_Name'], 'Processed Batches')

    ## Create Model Path Name
    Model_Name       = Model_Config["Model_Name"]
    Base_Model_Path  = os.path.join(CurDir, 'Saved Models', f'{Model_Name}')
    Base_Result_Path = os.path.join(CurDir, 'Performance Metrics', f'{Model_Name}')
    os.makedirs(Base_Model_Path,  exist_ok=True)
    os.makedirs(Base_Result_Path, exist_ok=True)

    Paths = {'Base_Model_Path': Base_Model_Path, 'Base_Result_Path': Base_Result_Path}

    ## Train Ensemble Members Sequentially, or Load Pre-Trained Models
    if Toggle_Config.get("Skip_Training", False):
        Device = DeclareDevice()
        Model_List, Loss, Metrics = LoadModels(Config, Paths, Norm_Parameters, Device)
    else:
        Model_List, Loss, Metrics = TrainModels(ProcessedDir, Config, Paths, Norm_Parameters, Input_Properties)

    ## Run Post Processing
    if Toggle_Config["Post_Process_Toggle"] is True:
        print(f"\n=== Running Post Processing After Training: {Model_Name} ===")

        Device  = DeclareDevice()
        Loaders = BuildLoaders(ProcessedDir, Data_Config, Input_Properties, Num_Workers=4)

        Ensembled_Result_Path = os.path.join(Base_Result_Path, 'Ensembled Model')
        os.makedirs(Ensembled_Result_Path, exist_ok=True)

        # Shared Arguments
        Ensemble_Args = dict(Loss=Loss, Metrics=Metrics, Device=Device,
                             Model_List=Model_List, Result_Path=Ensembled_Result_Path,
                             Output_Keys=Norm_Parameters['Keys']['Output_Keys'])

        # Evaluate Ensemble over each Split, followed by writing the per-sample CSV
        Splits = [('Train',      Loaders['Train_Eval'], False,                              False),
                  ('Validation', Loaders['Valid'],      False,                              False),
                  ('Test',       Loaders['Test'],       Toggle_Config['Visualize_Test_Case'],  Toggle_Config['Visualize_Results'])]

        for Split_Name, Split_Loader, Plot_Test_Case, Plot_Performance in Splits:
            ComputeResults(Dataset=Split_Loader, Split_Name=Split_Name,
                            Plot_Test_Case=Plot_Test_Case, Plot_Performance=Plot_Performance,
                            **Ensemble_Args)

        for Split_Name, Split_Loader, _, _ in Splits:
            WriteResultsToCSV(Dataset         = Split_Loader,
                              Model_List      = Model_List,
                              Device          = Device,
                              Result_Path     = Ensembled_Result_Path,
                              Split_Name      = Split_Name,
                              Ensemble_Toggle = Model_Config['N_Models'] > 1,
                              Metrics         = Metrics)

    return None