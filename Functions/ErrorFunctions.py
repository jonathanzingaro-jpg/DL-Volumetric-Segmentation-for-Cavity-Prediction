"""Configuration and dataset validation for the Blocks Consumed training pipeline.

    ValidatePaths        filesystem layout, before anything is loaded
    ValidateConfig       config file in isolation, fills defaults
    ValidateDatasetShape config against loaded tensors

Each is independent and raises ConfigError carrying every problem found.
Constants below mirror InitializeModel.py.
"""

from copy import deepcopy

SUPPORTED_MODELS      = ['UNet3D', 'Development Model']
SUPPORTED_LOSSES      = ['MSE', 'Dice+IoU', 'BCE', 'Soft_IoU', 'Tversky', 'Soft_IoU+BCE']
SUPPORTED_OPTIMIZERS  = ['Adam']
SUPPORTED_SCHEDULERS  = ['CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau', 'None']
SUPPORTED_TASK_TYPES    = ['Segmentation', 'Regression']
SUPPORTED_PROBLEM_TYPES = ['Incremental', 'Static', 'Dynamic']

REQUIRED_MODEL_KEYS     = ['Model_Type', 'In_Channels', 'Out_Channels']
REQUIRED_OPTIMIZER_KEYS = {'Adam': ['Learning_Rate', 'Weight_Decay', 'Betas', 'Eps']}
REQUIRED_SCHEDULER_KEYS = {'CosineAnnealingLR':            ['T_max', 'Min_Learning_Rate'],
                           'CosineAnnealingWarmRestarts':  ['T_0', 'T_mult'],
                           'ReduceLROnPlateau':            ['patience', 'factor', 'Min_Learning_Rate'],
                           'None':                         []}

# Applied when a key is absent
DEFAULTS = {'Model_Config':     {'Loss_Type': 'MSE', 'N_Epoch': 100,
                                 'N_Models': 1, 'Model_Name': 'Unnamed Model'},
            'Data_Config':      {'Noise_Toggle': False, 'Noise_Level': 0.0,
                                 'Augmentation_Toggle': False, 'Batch_Size': 32},
            'Optimizer_Config': {'Optimizer_Type': 'Adam', 'Weight_Decay': 0.0,
                                 'Betas': [0.9, 0.999], 'Eps': 1e-8},
            'Scheduler_Config': {'Scheduler_Type': 'None'},
            'Toggle_Config':    {'Post_Process_Toggle': False, 'Visualize_Results': False,
                                 'SHAP_Toggle': False, 'Multi_GPU': False,
                                 'Visualize_Test_Case': False, 'Skip_Training': False}}

# Keys required only when the named toggle is enabled.
CONDITIONAL_KEYS = {'Toggle_Config': {'Post_Process_Toggle': []}}

# Typo Detection -> Warning not Failure
KNOWN_KEYS = {'Model_Config':     ['In_Channels', 'Out_Channels', 'Model_Type', 'Loss_Type',
                                   'N_Epoch', 'N_Models', 'Model_Name', 'Temp_Scaling',
                                   'Task_Type', 'Problem_Type'],
              'Data_Config':      ['Dataset_Name', 'Noise_Toggle', 'Noise_Level',
                                   'Augmentation_Toggle', 'Batch_Size'],
              'Optimizer_Config': ['Optimizer_Type', 'Learning_Rate', 'Weight_Decay', 'Betas', 'Eps'],
              'Scheduler_Config': ['Scheduler_Type', 'T_0', 'T_mult', 'T_max', 'Min_Learning_Rate',
                                   'patience', 'factor'],
              'Toggle_Config':    ['Post_Process_Toggle', 'Visualize_Results', 'SHAP_Toggle', 'Multi_GPU',
                                   'Parallel_Ensemble', 'Visualize_Test_Case', 'Skip_Training']}

# Superseded Key Names
KEY_ALIASES = {'Toggle_Config': {'Multi-GPU': 'Multi_GPU', 'MultiGPU': 'Multi_GPU'},
               'Model_Config':  {'Epochs': 'N_Epoch', 'N_Epochs': 'N_Epoch'}}

DEFAULT_DOWNSAMPLE_FACTOR = 8
RAW_SAMPLE_LAYOUT         = ('Nx', 'Ny', 'Nz', 'Nt', 'Nc')
SPATIAL_AXIS_NAMES        = ('Nx', 'Ny', 'Nz')
SEQUENCE_AXIS_NAME        = 'Nt'
CHANNEL_AXIS_NAME         = 'Nc'
REQUIRED_INPUT_PROPERTIES = ['backfill_density']
SIGMOID_ACTIVATED_OUTPUT  = True
DOUBLE_SIGMOID_LOSSES     = ['BCE']
REQUIRED_DATASET_FILES    = ['params.pt']
INPUT_SAMPLE_NAMES        = ['Input', 'input', 'X', 'x', 'Inputs']
TARGET_SAMPLE_NAMES       = ['Target', 'target', 'Y', 'y', 'Label', 'Output', 'Targets']


class ConfigError(Exception):
    """Validation failure. Errors holds every problem found, in discovery order."""

    def __init__(self, Errors):
        self.Errors = list(Errors)
        Message = f"Configuration failed validation with {len(self.Errors)} error(s):\n"
        Message += "\n".join(f"  [{i}] {Error}" for i, Error in enumerate(self.Errors, start=1))
        super().__init__(Message)


# Helpers append to Errors
def _Check_Type(Section, Config, Key, Types, Type_Name, Errors):
    """Returns True if the value is present and of an acceptable type."""
    if Key not in Config:
        return False
    Value = Config[Key]
    # bool subclasses int; reject it where a number is expected
    if bool not in Types and isinstance(Value, bool):
        Errors.append(f"{Section}['{Key}'] must be {Type_Name}, got a boolean ({Value}).")
        return False
    if not isinstance(Value, tuple(Types)):
        Errors.append(f"{Section}['{Key}'] must be {Type_Name}, got {type(Value).__name__} ({Value!r}).")
        return False
    return True


def _Check_Number(Section, Config, Key, Errors, Minimum=None, Maximum=None,
                  Integer=False, Exclusive_Min=False, Exclusive_Max=False):
    """Type and range check for a numeric field. No-op if the key is absent."""
    Types      = [int] if Integer else [int, float]
    Type_Name  = "an integer" if Integer else "a number"
    if not _Check_Type(Section, Config, Key, Types, Type_Name, Errors):
        return

    Value = Config[Key]
    if Minimum is not None:
        if Exclusive_Min and Value <= Minimum:
            Errors.append(f"{Section}['{Key}'] must be greater than {Minimum}, got {Value}.")
        elif not Exclusive_Min and Value < Minimum:
            Errors.append(f"{Section}['{Key}'] must be >= {Minimum}, got {Value}.")
    if Maximum is not None:
        if Exclusive_Max and Value >= Maximum:
            Errors.append(f"{Section}['{Key}'] must be less than {Maximum}, got {Value}.")
        elif not Exclusive_Max and Value > Maximum:
            Errors.append(f"{Section}['{Key}'] must be <= {Maximum}, got {Value}.")


def _Check_Choice(Section, Config, Key, Options, Errors):
    if Key not in Config:
        return
    if Config[Key] not in Options:
        Errors.append(f"{Section}['{Key}'] = {Config[Key]!r} is not supported. Options are: {Options}.")


def _Check_Bool(Section, Config, Key, Errors):
    if Key in Config and not isinstance(Config[Key], bool):
        Errors.append(f"{Section}['{Key}'] must be true or false, got {Config[Key]!r}.")


def _Require(Section, Config, Keys, Errors, Reason=""):
    Suffix = f" {Reason}" if Reason else ""
    for Key in Keys:
        if Key not in Config:
            Errors.append(f"Missing required '{Key}' in {Section}.{Suffix}")


def _Unpack_Sample(Sample):
    """Extract (Input, Target) from a tuple/list, dict, or bare-tensor sample."""
    if isinstance(Sample, (list, tuple)):
        Input  = Sample[0] if len(Sample) > 0 else None
        Target = Sample[1] if len(Sample) > 1 else None
        return Input, Target

    if isinstance(Sample, dict):
        Input  = next((Sample[Name] for Name in INPUT_SAMPLE_NAMES  if Name in Sample), None)
        Target = next((Sample[Name] for Name in TARGET_SAMPLE_NAMES if Name in Sample), None)
        return Input, Target

    return Sample, None


def _Get_Shape(Tensor):
    """Return .shape as a tuple of ints for any array-like, or None if unavailable."""
    Shape = getattr(Tensor, 'shape', None)
    if Shape is None:
        return None
    try:
        return tuple(int(Dimension) for Dimension in Shape)
    except (TypeError, ValueError):
        return None


def ValidateConfig(Config, Strict=False, Verbose=True):
    """Validate a training configuration and fill in defaults.

    Args:
        Config  : dict of the five *_Config sections.
        Strict  : promote warnings to errors.
        Verbose : print warnings.

    Returns:
        (Validated_Config, Warnings) - a deep copy with defaults filled in.

    Raises:
        ConfigError listing every problem found.
    """
    Errors   = []
    Warnings = []

    if not isinstance(Config, dict):
        raise ConfigError([f"Config must be a dict, got {type(Config).__name__}."])

    Config = deepcopy(Config)

    ## Section Presence
    Sections = ['Model_Config', 'Data_Config', 'Optimizer_Config', 'Scheduler_Config', 'Toggle_Config']
    for Section in Sections:
        if Section not in Config:
            Config[Section] = {}
            Warnings.append(f"Section '{Section}' was missing - created empty and populated with defaults.")
        elif not isinstance(Config[Section], dict):
            Errors.append(f"Section '{Section}' must be a dict, got {type(Config[Section]).__name__}.")
            Config[Section] = {}

    for Section in Config:
        if Section not in Sections:
            Warnings.append(f"Unrecognised top-level section '{Section}' - it will be ignored.")

    ## Alias Migration
    for Section, Aliases in KEY_ALIASES.items():
        for Old_Key, New_Key in Aliases.items():
            if Old_Key in Config[Section]:
                if New_Key in Config[Section]:
                    Errors.append(f"{Section} contains both '{Old_Key}' and '{New_Key}'. Keep only '{New_Key}'.")
                else:
                    Config[Section][New_Key] = Config[Section].pop(Old_Key)
                    Warnings.append(f"{Section}['{Old_Key}'] renamed to '{New_Key}'.")

    ## Null Stripping
    for Section in Sections:
        for Key in [K for K, V in Config[Section].items() if V is None]:
            del Config[Section][Key]

    ## Typo Detection
    for Section, Keys in KNOWN_KEYS.items():
        for Key in Config[Section]:
            if Key not in Keys:
                Warnings.append(f"Unrecognised key {Section}['{Key}'] - it will be ignored.")

    ## Defaults, Recorded as Warnings
    for Section, Section_Defaults in DEFAULTS.items():
        for Key, Value in Section_Defaults.items():
            if Key not in Config[Section]:
                Config[Section][Key] = deepcopy(Value)
                Warnings.append(f"{Section}['{Key}'] was not specified - defaulting to {Value!r}.")

    Model_Config     = Config['Model_Config']
    Data_Config      = Config['Data_Config']
    Optimizer_Config = Config['Optimizer_Config']
    Scheduler_Config = Config['Scheduler_Config']
    Toggle_Config    = Config['Toggle_Config']

    ## Model_Config
    _Require('Model_Config', Model_Config, REQUIRED_MODEL_KEYS, Errors, "Required for model initialization.")
    _Check_Choice('Model_Config', Model_Config, 'Model_Type', SUPPORTED_MODELS, Errors)
    _Check_Choice('Model_Config', Model_Config, 'Loss_Type',  SUPPORTED_LOSSES, Errors)
    _Check_Choice('Model_Config', Model_Config, 'Task_Type',    SUPPORTED_TASK_TYPES, Errors)
    _Check_Choice('Model_Config', Model_Config, 'Problem_Type', SUPPORTED_PROBLEM_TYPES, Errors)
    _Check_Number('Model_Config', Model_Config, 'In_Channels',   Errors, Minimum=1, Integer=True)
    _Check_Number('Model_Config', Model_Config, 'Out_Channels',  Errors, Minimum=1, Integer=True)
    _Check_Number('Model_Config', Model_Config, 'N_Epoch',       Errors, Minimum=1, Integer=True)
    _Check_Number('Model_Config', Model_Config, 'N_Models', Errors, Minimum=1, Integer=True)
    _Check_Bool('Model_Config', Model_Config, 'Temp_Scaling', Errors)
    _Check_Type('Model_Config', Model_Config, 'Model_Name', [str], "a string", Errors)

    if SIGMOID_ACTIVATED_OUTPUT and Model_Config.get('Loss_Type') in DOUBLE_SIGMOID_LOSSES:
        Warnings.append(f"Model_Config['Loss_Type'] = {Model_Config['Loss_Type']!r} but UNet3D already "
                        f"applies torch.sigmoid to its output. Use the plain (non-logits) form of this "
                        f"loss, or the activation is applied twice.")


    ## Data_Config
    _Require('Data_Config', Data_Config, ['Dataset_Name'], Errors)
    _Check_Type('Data_Config', Data_Config, 'Dataset_Name', [str], "a string", Errors)
    _Check_Number('Data_Config', Data_Config, 'Batch_Size', Errors, Minimum=1, Integer=True)
    for Toggle in ['Noise_Toggle', 'Augmentation_Toggle']:
        _Check_Bool('Data_Config', Data_Config, Toggle, Errors)

    for Section_Name, Rules in CONDITIONAL_KEYS.items():
        Section_Config = Config[Section_Name]
        for Toggle, Dependent_Keys in Rules.items():
            if Section_Config.get(Toggle) is True:
                _Require(Section_Name, Section_Config, Dependent_Keys, Errors,
                         f"Required because {Toggle} is true.")
            else:
                for Key in Dependent_Keys:
                    if Key in Section_Config:
                        Warnings.append(f"{Section_Name}['{Key}'] is set but {Toggle} is false - it will be ignored.")

    _Check_Number('Data_Config', Data_Config, 'Noise_Level', Errors, Minimum=0.0, Maximum=1.0)

    # Train_Collate_Fn
    Noise_Toggle = Data_Config.get('Noise_Toggle')
    Noise_Level  = Data_Config.get('Noise_Level')

    if Noise_Toggle is True and Noise_Level == 0:
        Warnings.append("Data_Config['Noise_Toggle'] is true but 'Noise_Level' is 0 - no noise will "
                        "be applied. Raise the level or turn the toggle off.")

    elif Noise_Toggle is False and isinstance(Noise_Level, (int, float)) and Noise_Level > 0:
        Warnings.append(f"Data_Config['Noise_Level'] is {Noise_Level} but 'Noise_Toggle' is false - "
                        f"the level will be ignored and no noise applied.")

    ## Optimizer_Config
    _Check_Choice('Optimizer_Config', Optimizer_Config, 'Optimizer_Type', SUPPORTED_OPTIMIZERS, Errors)
    Optimizer_Type = Optimizer_Config.get('Optimizer_Type')
    if Optimizer_Type in REQUIRED_OPTIMIZER_KEYS:
        _Require('Optimizer_Config', Optimizer_Config, REQUIRED_OPTIMIZER_KEYS[Optimizer_Type], Errors,
                 f"Required for the {Optimizer_Type} optimizer.")

    _Check_Number('Optimizer_Config', Optimizer_Config, 'Learning_Rate', Errors, Minimum=0.0, Exclusive_Min=True)
    _Check_Number('Optimizer_Config', Optimizer_Config, 'Weight_Decay',  Errors, Minimum=0.0)
    _Check_Number('Optimizer_Config', Optimizer_Config, 'Eps',           Errors, Minimum=0.0, Exclusive_Min=True)

    if 'Betas' in Optimizer_Config:
        Betas = Optimizer_Config['Betas']
        if not isinstance(Betas, (list, tuple)) or len(Betas) != 2:
            Errors.append(f"Optimizer_Config['Betas'] must be a list of two values, got {Betas!r}.")
        else:
            for i, Beta in enumerate(Betas):
                if not isinstance(Beta, (int, float)) or isinstance(Beta, bool):
                    Errors.append(f"Optimizer_Config['Betas'][{i}] must be a number, got {Beta!r}.")
                elif not 0.0 <= Beta < 1.0:
                    Errors.append(f"Optimizer_Config['Betas'][{i}] must be in [0, 1), got {Beta}.")

    if isinstance(Optimizer_Config.get('Learning_Rate'), (int, float)) and Optimizer_Config['Learning_Rate'] > 1.0:
        Warnings.append(f"Optimizer_Config['Learning_Rate'] = {Optimizer_Config['Learning_Rate']} is unusually large.")

    ## Scheduler_Config
    _Check_Choice('Scheduler_Config', Scheduler_Config, 'Scheduler_Type', SUPPORTED_SCHEDULERS, Errors)
    Scheduler_Type = Scheduler_Config.get('Scheduler_Type')
    if Scheduler_Type in REQUIRED_SCHEDULER_KEYS:
        _Require('Scheduler_Config', Scheduler_Config, REQUIRED_SCHEDULER_KEYS[Scheduler_Type], Errors,
                 f"Required for the {Scheduler_Type} scheduler.")

    _Check_Number('Scheduler_Config', Scheduler_Config, 'T_0',    Errors, Minimum=1, Integer=True)
    _Check_Number('Scheduler_Config', Scheduler_Config, 'T_mult', Errors, Minimum=1, Integer=True)
    _Check_Number('Scheduler_Config', Scheduler_Config, 'T_max',  Errors, Minimum=1, Integer=True)
    _Check_Number('Scheduler_Config', Scheduler_Config, 'patience', Errors, Minimum=0, Integer=True)
    _Check_Number('Scheduler_Config', Scheduler_Config, 'factor', Errors, Minimum=0.0, Maximum=1.0, Exclusive_Min=True)
    _Check_Number('Scheduler_Config', Scheduler_Config, 'Min_Learning_Rate', Errors, Minimum=0.0)

    # Learning Rate Check
    Learning_Rate     = Optimizer_Config.get('Learning_Rate')
    Min_Learning_Rate = Scheduler_Config.get('Min_Learning_Rate')
    if isinstance(Learning_Rate, (int, float)) and isinstance(Min_Learning_Rate, (int, float)):
        if Min_Learning_Rate > Learning_Rate:
            Errors.append(f"Scheduler_Config['Min_Learning_Rate'] ({Min_Learning_Rate}) exceeds "
                          f"Optimizer_Config['Learning_Rate'] ({Learning_Rate}).")

    # Warm Restart Behaviour
    N_Epoch = Model_Config.get('N_Epoch')
    if Scheduler_Type == 'CosineAnnealingWarmRestarts' and isinstance(N_Epoch, int):
        T_0    = Scheduler_Config.get('T_0')
        T_mult = Scheduler_Config.get('T_mult')

        if isinstance(T_0, int) and T_0 > N_Epoch:
            Warnings.append(f"Scheduler_Config['T_0'] ({T_0}) exceeds N_Epoch ({N_Epoch}) - "
                            f"no warm restart will occur.")

        elif isinstance(T_0, int) and isinstance(T_mult, int) and T_0 > 0 and T_mult >= 1:

            Boundaries, Total, Cycle = [], 0, T_0
            while Total < N_Epoch and len(Boundaries) < 64:
                Total += Cycle
                Boundaries.append(Total)
                Cycle *= T_mult
            if N_Epoch not in Boundaries:
                Nearest = [B for B in Boundaries if B <= N_Epoch][-1:] + [B for B in Boundaries if B > N_Epoch][:1]
                Warnings.append(f"N_Epoch ({N_Epoch}) does not land on a warm restart boundary "
                                f"{Boundaries[:6]} - training stops mid-cycle at a raised learning rate. "
                                f"Nearest boundaries: {Nearest}.")

    if Scheduler_Type == 'ReduceLROnPlateau' and isinstance(N_Epoch, int):
        Patience = Scheduler_Config.get('patience')
        if isinstance(Patience, int) and Patience >= N_Epoch:
            Warnings.append(f"Scheduler_Config['patience'] ({Patience}) is >= N_Epoch ({N_Epoch}) - "
                            f"the learning rate will never be reduced.")

    ## Toggle_Config
    for Toggle in ['Post_Process_Toggle', 'Visualize_Results', 'SHAP_Toggle', 'Multi_GPU', 'Parallel_Ensemble',
                   'Visualize_Test_Case', 'Skip_Training']:
        _Check_Bool('Toggle_Config', Toggle_Config, Toggle, Errors)

    # Ensemble Check
    N_Models = Model_Config.get('N_Models')

    # SHAP Cost Scales with Ensemble Size and Input Volume
    if Toggle_Config.get('SHAP_Toggle') is True and isinstance(N_Models, int) and N_Models > 1:
        Warnings.append(f"Toggle_Config['SHAP_Toggle'] is true across {N_Models} ensemble members - "
                        f"attribution on volumetric inputs is slow and will dominate the run time.")

    # DataParallel Shards the Batch across Devices
    if Toggle_Config.get('Multi_GPU') is True:
        Batch_Size = Data_Config.get('Batch_Size')
        if isinstance(Batch_Size, int) and Batch_Size % 2 != 0:
            Warnings.append(f"Data_Config['Batch_Size'] ({Batch_Size}) is odd while Multi_GPU is true. "
                            f"DataParallel splits the batch across devices, so an uneven split leaves "
                            f"one device with a smaller sub-batch and skews BatchNorm statistics.")

    # Skipping Training 
    if Toggle_Config.get('Skip_Training') is True:
        Warnings.append("Toggle_Config['Skip_Training'] is true - Optimizer_Config, Scheduler_Config, "
                        "and Model_Config['N_Epoch'] will be ignored since no training will run. "
                        "Make sure a pretrained checkpoint is available to load instead.")
        if Toggle_Config.get('SHAP_Toggle') is True or Toggle_Config.get('Visualize_Test_Case') is True:
            Warnings.append("Toggle_Config['Skip_Training'] is true alongside SHAP_Toggle/Visualize_Test_Case - "
                            "these will run against whatever weights are loaded, not a freshly trained model.")

    if Toggle_Config.get('Visualize_Test_Case') is True and Toggle_Config.get('Skip_Training') is not True \
            and not isinstance(N_Epoch, int):
        Warnings.append("Toggle_Config['Visualize_Test_Case'] is true but Model_Config['N_Epoch'] is unset - "
                        "cannot confirm a trained model will exist to visualise.")

    ## Report
    if Strict and Warnings:
        Errors.extend(f"(strict) {Warning}" for Warning in Warnings)

    if Errors:
        raise ConfigError(Errors)

    if Verbose:
        for Warning in Warnings:
            print(f"[Config] Warning: {Warning}")

    return Config, Warnings


def ValidateDatasetShape(Config, Norm_Parameters, Datasets,
                         Downsample_Factor=DEFAULT_DOWNSAMPLE_FACTOR,
                         Raw_Layout=RAW_SAMPLE_LAYOUT, Verbose=True):
    """Check the model geometry in the config against the loaded data.

    Args:
        Config           : the validated config returned by ValidateConfig.
        Norm_Parameters  : the params.pt dict, expects Norm_Parameters['Keys'].
        Datasets         : a Dataset/DataLoader or a dict of them. Prefer DataLoaders;
                           collate-time transforms are already applied.
        Downsample_Factor: 2 ** (number of Down3d stages). UNet3D has 3, so 8.
        Raw_Layout       : axis names for one sample.
        Verbose          : print warnings and a confirmation line.

    Returns:
        Warnings (list of str)

    Raises:
        ConfigError listing every mismatch found.
    """
    Errors   = []
    Warnings = []

    Model_Config = Config.get('Model_Config', {})
    Data_Config  = Config.get('Data_Config', {})
    In_Channels  = Model_Config.get('In_Channels')
    Out_Channels = Model_Config.get('Out_Channels')
    Model_Type   = Model_Config.get('Model_Type')

    if CHANNEL_AXIS_NAME not in Raw_Layout:
        raise ConfigError([f"Raw_Layout {Raw_Layout} must contain '{CHANNEL_AXIS_NAME}'."])

    ## Norm_Parameters Keys
    Keys        = Norm_Parameters.get('Keys', {}) if isinstance(Norm_Parameters, dict) else {}
    Input_Keys  = Keys.get('Input_Keys')
    Output_Keys = Keys.get('Output_Keys')

    if Input_Keys is None:
        Errors.append("Norm_Parameters['Keys']['Input_Keys'] is missing - cannot verify input channels.")
    else:
        if len(Input_Keys) != In_Channels:
            Errors.append(f"Model_Config['In_Channels'] = {In_Channels} but the dataset declares "
                          f"{len(Input_Keys)} input properties in Norm_Parameters['Keys']['Input_Keys']: "
                          f"{list(Input_Keys)}.")
        if len(set(Input_Keys)) != len(Input_Keys):
            Errors.append(f"Norm_Parameters['Keys']['Input_Keys'] contains duplicates: {list(Input_Keys)}.")

        for Property in REQUIRED_INPUT_PROPERTIES:
            if Property not in Input_Keys:
                Errors.append(f"UNet3D requires '{Property}' in Norm_Parameters['Keys']['Input_Keys'] "
                              f"but it is not present. Available: {list(Input_Keys)}.")

    if Output_Keys is None:
        Warnings.append("Norm_Parameters['Keys']['Output_Keys'] is missing - cannot verify output channels.")
    elif len(Output_Keys) != Out_Channels:
        Errors.append(f"Model_Config['Out_Channels'] = {Out_Channels} but the dataset declares "
                      f"{len(Output_Keys)} output properties in Norm_Parameters['Keys']['Output_Keys']: "
                      f"{list(Output_Keys)}.")

    ## Per-Split tensor Shapes
    if not isinstance(Datasets, dict):
        Datasets = {'Dataset': Datasets}

    Reference_Spatial = None
    Reference_Split   = None

    for Split_Name, Dataset in Datasets.items():

        Is_Loader = hasattr(Dataset, 'dataset')     # DataLoader wraps its Dataset
        Layout    = ('B',) + tuple(Raw_Layout) if Is_Loader else tuple(Raw_Layout)
        Kind      = "batch" if Is_Loader else "sample"

        # Length in Samples
        try:
            Length = len(Dataset.dataset) if Is_Loader else len(Dataset)
        except TypeError:
            Warnings.append(f"{Split_Name} dataset has no length - skipping shape checks for this split.")
            continue

        if Length == 0:
            Errors.append(f"{Split_Name} dataset is empty.")
            continue

        Batch_Size = Data_Config.get('Batch_Size')
        if isinstance(Batch_Size, int) and Batch_Size > Length:
            Warnings.append(f"Batch_Size ({Batch_Size}) exceeds the {Split_Name} dataset size ({Length}) - "
                            f"every epoch will be a single partial batch.")

        # DataLoader Yields a Collated Batch 
        try:
            Sample = next(iter(Dataset)) if Is_Loader else Dataset[0]
        except Exception as Error:
            Warnings.append(f"Could not read a {Kind} from the {Split_Name} split ({Error}) - "
                            f"skipping shape checks for this split.")
            continue

        Input, Target = _Unpack_Sample(Sample)
        Input_Shape   = _Get_Shape(Input)
        Target_Shape  = _Get_Shape(Target)

        if Input_Shape is None:
            Warnings.append(f"{Split_Name} {Kind} input has no .shape attribute "
                            f"({type(Input).__name__}) - skipping shape checks for this split.")
            continue

        if len(Input_Shape) != len(Layout):
            Errors.append(f"{Split_Name} {Kind} input has shape {Input_Shape} ({len(Input_Shape)} dims) "
                          f"but the expected layout is {Layout} ({len(Layout)} dims).")
            continue

        Axis_Of         = {Name: Index for Index, Name in enumerate(Layout)}
        Sample_Channels = Input_Shape[Axis_Of[CHANNEL_AXIS_NAME]]
        Spatial         = tuple(Input_Shape[Axis_Of[Name]] for Name in SPATIAL_AXIS_NAMES if Name in Axis_Of)
        Sequence        = Input_Shape[Axis_Of[SEQUENCE_AXIS_NAME]] if SEQUENCE_AXIS_NAME in Axis_Of else None

        ## Channels
        if Sample_Channels != In_Channels:
            Errors.append(f"Model_Config['In_Channels'] = {In_Channels} but the {Split_Name} split "
                          f"provides {Sample_Channels} channels at axis "
                          f"{Axis_Of[CHANNEL_AXIS_NAME]} (shape {Input_Shape}, layout {Layout}).")

        ## Sequence Length
        if Sequence is not None and Sequence != 1:
            Errors.append(f"{Split_Name} split has Nt = {Sequence}, but UNet3D.forward uses "
                          f"x.squeeze(-2), which only removes that axis when it is 1. With Nt > 1 the "
                          f"following permute(0, 4, 1, 2, 3) receives a 6D tensor and will fail.")

        ## Spatial Geometry
        if Model_Type == 'UNet3D' and Downsample_Factor > 1:
            Bad_Dimensions = [(Name, Size) for Name, Size in zip(SPATIAL_AXIS_NAMES, Spatial)
                              if Size % Downsample_Factor != 0]
            if Bad_Dimensions:
                Detail = ", ".join(f"{Name}={Size}" for Name, Size in Bad_Dimensions)
                Errors.append(f"{Split_Name} volume {Spatial} is not divisible by {Downsample_Factor} "
                              f"({Detail}). UNet3D pools 3 times, so each spatial dim must survive "
                              f"3 halvings - pad or crop the volumes.")

        if Reference_Spatial is None:
            Reference_Spatial, Reference_Split = Spatial, Split_Name
        elif Spatial != Reference_Spatial:
            Errors.append(f"{Split_Name} volume {Spatial} differs from {Reference_Split} "
                          f"{Reference_Spatial}. All splits must share the same geometry.")

        ## Target -> UNet3D Returns (B, Nx, Ny, Nz, 1, Nc_out)
        if Target is None:
            Warnings.append(f"{Split_Name} {Kind} has no target - cannot verify Out_Channels.")
        elif Target_Shape is None:
            Warnings.append(f"{Split_Name} {Kind} target has no .shape attribute "
                            f"({type(Target).__name__}) - cannot verify Out_Channels.")
        elif len(Target_Shape) != len(Layout):
            Errors.append(f"{Split_Name} {Kind} target has shape {Target_Shape} ({len(Target_Shape)} "
                          f"dims) but the model returns {len(Layout)} dims in layout {Layout}.")
        else:
            Target_Channels = Target_Shape[Axis_Of[CHANNEL_AXIS_NAME]]
            Target_Spatial  = tuple(Target_Shape[Axis_Of[Name]] for Name in SPATIAL_AXIS_NAMES
                                    if Name in Axis_Of)

            if Target_Channels != Out_Channels:
                Errors.append(f"Model_Config['Out_Channels'] = {Out_Channels} but the {Split_Name} split "
                              f"provides {Target_Channels} target channels (shape {Target_Shape}).")

            if Target_Spatial != Spatial:
                Errors.append(f"{Split_Name} target volume {Target_Spatial} does not match the input "
                              f"volume {Spatial}.")

    if Errors:
        raise ConfigError(Errors)

    if Verbose:
        for Warning in Warnings:
            print(f"[Shape] Warning: {Warning}")
        if Reference_Spatial is not None:
            print(f"[Shape] OK - {In_Channels} in / {Out_Channels} out, volume {Reference_Spatial}, "
                  f"layout {tuple(Raw_Layout)}.")

    return Warnings


def ValidatePaths(CurDir, Config, Splits=('Train_Dataset', 'Valid_Dataset', 'Test_Dataset'),
                  Check_Writable=True, Verbose=True):
    """Check the dataset folder, split files and output folders before anything loads.

    Args:
        CurDir         : project root, the same value passed to Main.
        Config         : the validated config returned by ValidateConfig.
        Splits         : split file stems expected inside the dataset folder.
        Check_Writable : probe the output folders by writing and deleting a file.
        Verbose        : print warnings and a confirmation line.

    Returns:
        Warnings (list of str)

    Raises:
        ConfigError listing every problem found.
    """
    import os
    import glob

    Errors   = []
    Warnings = []

    Model_Config = Config.get('Model_Config', {})
    Data_Config  = Config.get('Data_Config', {})
    Dataset_Name = Data_Config.get('Dataset_Name')
    Model_Name   = Model_Config.get('Model_Name')

    if not Dataset_Name:
        raise ConfigError(["Data_Config['Dataset_Name'] is missing - cannot locate the dataset."])

    ## Dataset Folder
    Dataset_Path = os.path.join(CurDir, 'Dataset', Dataset_Name)
    if not os.path.isdir(Dataset_Path):
        Siblings = sorted(os.listdir(os.path.join(CurDir, 'Dataset'))) \
            if os.path.isdir(os.path.join(CurDir, 'Dataset')) else []
        Detail = f" Available datasets: {Siblings}." if Siblings else ""
        raise ConfigError([f"No dataset folder at {Dataset_Path}.{Detail}"])

    ## Required Files
    for Filename in REQUIRED_DATASET_FILES:
        if not os.path.isfile(os.path.join(Dataset_Path, Filename)):
            Errors.append(f"Missing '{Filename}' in {Dataset_Path}.")

    ## Split Files
    for Split in Splits:
        Matches = glob.glob(os.path.join(Dataset_Path, Split + '.*')) + \
                  glob.glob(os.path.join(Dataset_Path, Split))
        if not Matches:
            Present = sorted(os.path.basename(P) for P in glob.glob(os.path.join(Dataset_Path, '*')))
            Errors.append(f"No file matching '{Split}' in {Dataset_Path}. Present: {Present}.")

    ## Output Folders
    if not Model_Name:
        Warnings.append("Model_Config['Model_Name'] is empty - output folders cannot be checked.")
    else:
        for Folder in ['Saved Models', 'Performance Metrics']:
            Path = os.path.join(CurDir, Folder, Model_Name)

            # Populated Output Folder > Run Overwrites Previous Results
            if os.path.isdir(Path) and os.listdir(Path):
                Warnings.append(f"{Path} already exists and is not empty - this run will overwrite "
                                f"previous results for '{Model_Name}'.")

            if not Check_Writable:
                continue
            try:
                os.makedirs(Path, exist_ok=True)
                Probe = os.path.join(Path, '.write_probe')
                with open(Probe, 'w') as File:
                    File.write('ok')
                os.remove(Probe)
            except OSError as Error:
                Errors.append(f"Cannot write to {Path} ({Error}).")

    if Errors:
        raise ConfigError(Errors)

    if Verbose:
        for Warning in Warnings:
            print(f"[Paths] Warning: {Warning}")
        print(f"[Paths] OK - dataset at {Dataset_Path}, outputs writable.")

    return Warnings