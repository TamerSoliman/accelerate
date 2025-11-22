# Integration 1: Tracker Base Class & Implementation Pattern

## Overview

Accelerate provides unified experiment tracking across **9 different logging platforms** through a common `GeneralTracker` base class. This design allows users to log metrics to TensorBoard, WandB, MLflow, or any combination simultaneously using the same API.

**File Location:** `src/accelerate/tracking.py:101-180` (GeneralTracker base class)

**Supported Trackers:**
1. **TensorBoard** - Local file-based logging (`tracking.py:182-295`)
2. **Weights & Biases (WandB)** - Cloud-based experiment tracking (`tracking.py:297-429`)
3. **Trackio** - Gradio's tracking system (`tracking.py:431-506`)
4. **Comet ML** - Cloud-based ML platform (`tracking.py:508-600`)
5. **Aim** - Open-source experiment tracking (`tracking.py:602-703`)
6. **MLflow** - Open-source ML lifecycle platform (`tracking.py:705-910`)
7. **ClearML** - ML/DL experiment management (`tracking.py:912-1068`)
8. **DVCLive** - Data Version Control's live logging (`tracking.py:1070-1156`)
9. **SwanLab** - Chinese ML experiment tracking platform (`tracking.py:1158-1256`)

---

## Architecture Pattern

### The Strategy Pattern

Accelerate uses the **Strategy Pattern** for tracker implementations:
- **Base class (`GeneralTracker`)** defines the interface
- **Concrete classes** (`TensorBoardTracker`, `WandBTracker`, etc.) implement platform-specific logic
- **Accelerator** uses trackers through the common interface

```
┌────────────────────┐
│  Accelerator       │
│  .init_trackers()  │
│  .log()            │
└────────┬───────────┘
         │ uses
         ▼
┌────────────────────┐
│  GeneralTracker    │  ← Base class (abstract interface)
│  + start()         │
│  + log()           │
│  + store_init...() │
│  + finish()        │
└────────┬───────────┘
         │ extends
    ┌────┴─────┬──────────┬──────────┬─────────┐
    ▼          ▼          ▼          ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  ...
│TensorB.│ │ WandB  │ │Comet ML│ │MLflow  │
│Tracker │ │Tracker │ │Tracker │ │Tracker │
└────────┘ └────────┘ └────────┘ └────────┘
```

---

## Annotated Base Class: `GeneralTracker`

**Location:** `src/accelerate/tracking.py:101-180`

```python
class GeneralTracker:
    """
    A base Tracker class to be used for all logging integration implementations.

    # WHAT: This is the abstract base class that defines the contract
    # all tracker implementations must follow.

    # WHY: Provides a unified API so Accelerator can work with any tracker
    # without knowing implementation details. Enables multi-tracker logging.

    # HOW: Uses duck typing - subclasses must implement required attributes
    # (`name`, `requires_logging_directory`, `tracker` property) and methods
    # (`start`, `store_init_configuration`, `log`, `finish`).
    """

    # --- Attribute 1: Main Process Only ---
    main_process_only = True
    # WHAT: Class attribute controlling whether logging happens only on main process
    # WHY: In distributed training, you usually want only ONE process to log metrics
    #      to avoid duplicate entries. However, some trackers (WandB, SwanLab, Trackio)
    #      set this to False because they handle distributed logging internally.
    # HOW: The @on_main_process decorator checks this attribute and conditionally
    #      executes methods based on PartialState().on_main_process

    def __init__(self, _blank=False):
        # WHAT: Validates that subclass implements required attributes
        # WHY: Python doesn't have abstract base classes enforcement by default,
        #      so we manually check at init time
        # HOW: Uses hasattr() and dir() to verify presence of required attributes

        if not _blank:
            err = ""

            # Check 1: name attribute
            if not hasattr(self, "name"):
                err += "`name`"
            # WHAT: String representation like "tensorboard", "wandb"
            # WHY: Used to identify tracker type in logs and error messages
            # EXAMPLE: TensorBoardTracker.name = "tensorboard"

            # Check 2: requires_logging_directory attribute
            if not hasattr(self, "requires_logging_directory"):
                if len(err) > 0:
                    err += ", "
                err += "`requires_logging_directory`"
            # WHAT: Boolean indicating if tracker needs a local directory for logs
            # WHY: TensorBoard/Aim store logs locally, WandB/MLflow use cloud
            # EXAMPLE: TensorBoardTracker.requires_logging_directory = True
            #          WandBTracker.requires_logging_directory = False

            # Check 3: tracker property (as a @property, so check in dir())
            if "tracker" not in dir(self):
                if len(err) > 0:
                    err += ", "
                err += "`tracker`"
            # WHAT: Property that returns the underlying tracker object
            # WHY: Allows users to access native API for advanced features
            # EXAMPLE: accelerator.get_tracker("wandb").run.watch(model)
            #          Returns the wandb.Run object

            if len(err) > 0:
                raise NotImplementedError(
                    f"The implementation for this tracker class is missing the following "
                    f"required attributes. Please define them in the class definition: "
                    f"{err}"
                )

    # --- Method 1: Lazy Initialization ---
    def start(self):
        """
        Lazy initialization of the tracker inside Accelerator to avoid initializing
        PartialState before InitProcessGroupKwargs.

        # WHAT: Performs actual initialization of the tracking backend
        # WHY: Delayed until after distributed setup is complete
        # HOW: Subclasses import and initialize their respective libraries here
        #      (e.g., import wandb; wandb.init())

        # TIMING: Called by Accelerator.init_trackers(), which is typically called
        # after prepare() so distributed environment is fully set up
        """
        pass

    # --- Method 2: Store Configuration ---
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Implementations should use the
        experiment configuration functionality of a tracking API.

        # WHAT: Logs hyperparameters at the start of training
        # WHY: Records experiment configuration for reproducibility
        # HOW: Each tracker maps to its native hyperparameter API:
        #   - TensorBoard: writer.add_hparams(values)
        #   - WandB: wandb.config.update(values)
        #   - MLflow: mlflow.log_params(values)
        #   - CometML: experiment.log_parameters(values)

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs.
                The values need to have type `bool`, `str`, `float`, `int`, or `None`.

        # EXAMPLE:
        # accelerator.init_trackers("my_project")
        # config = {"learning_rate": 1e-4, "batch_size": 32, "model": "bert-base"}
        # accelerator.trackers[0].store_init_configuration(config)
        """
        pass

    # --- Method 3: Log Metrics ---
    def log(self, values: dict, step: Optional[int], **kwargs):
        """
        Logs `values` to the current run. Base `log` implementations of a tracking API
        should go in here, along with special behavior for the `step` parameter.

        # WHAT: Logs metrics during training
        # WHY: Track training progress, validation scores, etc.
        # HOW: Each tracker maps to its native logging API:
        #   - TensorBoard: writer.add_scalar(k, v, global_step=step)
        #   - WandB: run.log(values, step=step)
        #   - MLflow: mlflow.log_metrics(values, step=step)
        #   - CometML: experiment.log_metric(k, v, step=step)

        Args:
            values (Dictionary `str` to `str`, `float`, or `int`):
                Values to be logged as key-value pairs. The values need to have type
                `str`, `float`, or `int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.

        # EXAMPLE:
        # for step, batch in enumerate(train_dataloader):
        #     loss = train_step(batch)
        #     accelerator.log({"train/loss": loss}, step=step)
        """
        pass

    # --- Method 4: Cleanup ---
    def finish(self):
        """
        Should run any finalizing functions within the tracking API. If the API should
        not have one, just don't overwrite that method.

        # WHAT: Clean up tracker resources
        # WHY: Close files, upload final data, mark run as complete
        # HOW: Each tracker calls its cleanup method:
        #   - TensorBoard: writer.close()
        #   - WandB: run.finish()
        #   - MLflow: mlflow.end_run()
        #   - CometML: experiment.end()

        # TIMING: Called automatically by Accelerator.end_training() or manually
        # via accelerator.end_training()
        """
        pass
```

---

## The `@on_main_process` Decorator

**Location:** `src/accelerate/tracking.py:77-93`

This decorator ensures logging only happens on the main process in distributed training.

```python
def on_main_process(function):
    """
    Decorator to selectively run the decorated function on the main process only
    based on the `main_process_only` attribute in a class.

    # WHAT: Conditional execution decorator
    # WHY: Prevents duplicate logging in multi-GPU training
    # HOW: Checks class attribute at runtime, wraps function with PartialState check

    Checks at function execution rather than initialization time, not triggering
    the initialization of the `PartialState`.
    """

    @wraps(function)
    def execute_on_main_process(self, *args, **kwargs):
        # Check if this tracker wants main-process-only behavior
        if getattr(self, "main_process_only", False):
            # WHAT: Use PartialState to check if we're on main process
            # WHY: PartialState knows distributed rank without full Accelerator init
            # HOW: Returns a wrapped function that only executes on process 0
            return PartialState().on_main_process(function)(self, *args, **kwargs)
        else:
            # WHAT: Execute on all processes
            # WHY: Some trackers (WandB, SwanLab) handle distributed internally
            # HOW: Direct function call without rank checking
            return function(self, *args, **kwargs)

    return execute_on_main_process
```

**Key Insight:** The decorator checks `main_process_only` at **runtime** (when method is called), not at initialization time. This avoids triggering `PartialState()` initialization too early.

---

## Tracker Implementation Examples

### Example 1: TensorBoard (File-Based, Requires Directory)

**Location:** `src/accelerate/tracking.py:182-295`

```python
class TensorBoardTracker(GeneralTracker):
    """
    A `Tracker` class that supports `tensorboard`. Should be initialized at the
    start of your script.

    # WHAT: Tracker for local TensorBoard logging
    # WHY: TensorBoard is the most common PyTorch logging tool
    # HOW: Wraps torch.utils.tensorboard.SummaryWriter
    """

    # Required attributes from GeneralTracker
    name = "tensorboard"
    requires_logging_directory = True  # Stores logs locally

    def __init__(self, run_name: str, logging_dir: Union[str, os.PathLike], **kwargs):
        super().__init__()
        # WHAT: Store initialization params for lazy loading
        # WHY: Actual SummaryWriter created in start(), not __init__()
        # HOW: Save params for later use
        self.run_name = run_name
        self.logging_dir_param = logging_dir
        self.init_kwargs = kwargs

    @on_main_process  # Only main process writes to TensorBoard
    def start(self):
        # WHAT: Import TensorBoard and create writer
        # WHY: Delayed import until distributed setup is complete
        # HOW: Try torch.utils.tensorboard first, fallback to tensorboardX
        try:
            from torch.utils import tensorboard
        except ModuleNotFoundError:
            import tensorboardX as tensorboard

        # Create logging directory: {logging_dir}/{run_name}/
        self.logging_dir = os.path.join(self.logging_dir_param, self.run_name)
        self.writer = tensorboard.SummaryWriter(self.logging_dir, **self.init_kwargs)

        logger.debug(f"Initialized TensorBoard project {self.run_name} logging to {self.logging_dir}")

    @property
    def tracker(self):
        # WHAT: Return underlying SummaryWriter
        # WHY: Allows users to access native TensorBoard API
        # EXAMPLE: accelerator.get_tracker("tensorboard").add_graph(model, input)
        return self.writer

    @on_main_process
    def store_init_configuration(self, values: dict):
        # WHAT: Log hyperparameters to TensorBoard
        # HOW: Uses add_hparams() and saves YAML file for backup
        self.writer.add_hparams(values, metric_dict={})
        self.writer.flush()

        # Also save as YAML file for easy access
        project_run_name = time.time()
        dir_name = os.path.join(self.logging_dir, str(project_run_name))
        os.makedirs(dir_name, exist_ok=True)
        with open(os.path.join(dir_name, "hparams.yml"), "w") as outfile:
            yaml.dump(values, outfile)

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        # WHAT: Log metrics with type handling
        # WHY: TensorBoard has different methods for scalars, text, and dicts
        # HOW: Type dispatch based on value type

        values = listify(values)  # Ensure values is a flat dict
        for k, v in values.items():
            if isinstance(v, (int, float)):
                # Scalar metrics (loss, accuracy, etc.)
                self.writer.add_scalar(k, v, global_step=step, **kwargs)
            elif isinstance(v, str):
                # Text logging (model predictions, errors, etc.)
                self.writer.add_text(k, v, global_step=step, **kwargs)
            elif isinstance(v, dict):
                # Multiple related scalars (e.g., per-class accuracy)
                self.writer.add_scalars(k, v, global_step=step, **kwargs)
        self.writer.flush()

    @on_main_process
    def finish(self):
        # WHAT: Close TensorBoard writer
        # WHY: Flush remaining data, release file handles
        self.writer.close()
```

**Key Characteristics:**
- **Requires logging directory**: Yes (stores event files locally)
- **Main process only**: Yes (file-based, multi-process writes would conflict)
- **Type handling**: Different methods for scalars, text, and scalar groups
- **Backup**: Saves hyperparameters as YAML in addition to TensorBoard format

---

### Example 2: Weights & Biases (Cloud-Based, No Directory)

**Location:** `src/accelerate/tracking.py:297-429`

```python
class WandBTracker(GeneralTracker):
    """
    A `Tracker` class that supports `wandb`. Should be initialized at the start of
    your script.

    # WHAT: Tracker for Weights & Biases cloud platform
    # WHY: WandB provides advanced visualizations, model versioning, and collaboration
    # HOW: Wraps wandb.init() and wandb.log()
    """

    name = "wandb"
    requires_logging_directory = False  # Cloud-based, no local files
    main_process_only = False  # WandB handles distributed internally

    def __init__(self, run_name: str, **kwargs):
        super().__init__()
        self.run_name = run_name
        self.init_kwargs = kwargs

    @on_main_process
    def start(self):
        import wandb

        # WHAT: Initialize WandB run
        # WHY: Creates experiment on WandB servers
        # HOW: wandb.init() returns a Run object
        self.run = wandb.init(project=self.run_name, **self.init_kwargs)

        logger.debug(f"Initialized WandB project {self.run_name}")

    @property
    def tracker(self):
        # Return wandb.Run object for advanced features
        return self.run

    @on_main_process
    def store_init_configuration(self, values: dict):
        import wandb

        # WHAT: Handle offline vs online mode differently
        # WHY: Offline mode requires config at init time
        # HOW: Check WANDB_MODE environment variable

        if os.environ.get("WANDB_MODE") == "offline":
            # In offline mode, restart wandb with config included
            if hasattr(self, "run") and self.run:
                self.run.finish()

            init_kwargs = self.init_kwargs.copy()
            init_kwargs["config"] = values
            self.run = wandb.init(project=self.run_name, **init_kwargs)
        else:
            # In online mode, update config dynamically
            wandb.config.update(values, allow_val_change=True)

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        # WHAT: Log metrics to WandB
        # WHY: Simple API - WandB handles all types automatically
        # HOW: Single method call, WandB infers types
        self.run.log(values, step=step, **kwargs)

    @on_main_process
    def log_images(self, values: dict, step: Optional[int] = None, **kwargs):
        # WHAT: Log images to WandB
        # WHY: WandB provides rich image visualizations
        # HOW: Wrap images in wandb.Image objects
        import wandb

        for k, v in values.items():
            self.log({k: [wandb.Image(image) for image in v]}, step=step, **kwargs)

    @on_main_process
    def log_table(
        self,
        table_name: str,
        columns: Optional[list[str]] = None,
        data: Optional[list[list[Any]]] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
        **kwargs,
    ):
        # WHAT: Log structured data tables
        # WHY: WandB Tables support rich data types (images, audio, etc.)
        # HOW: Create wandb.Table and log like any other value
        import wandb

        values = {table_name: wandb.Table(columns=columns, data=data, dataframe=dataframe)}
        self.log(values, step=step, **kwargs)

    @on_main_process
    def finish(self):
        # WHAT: Mark WandB run as complete
        # WHY: Uploads remaining data, updates run status
        self.run.finish()
```

**Key Characteristics:**
- **Requires logging directory**: No (cloud-based)
- **Main process only**: **No** (WandB handles distributed logging internally)
- **Special handling**: Offline mode requires config at init
- **Advanced features**: Images, tables, model watching

---

### Example 3: MLflow (Flexible, Environment-Aware)

**Location:** `src/accelerate/tracking.py:705-910`

```python
class MLflowTracker(GeneralTracker):
    """
    A `Tracker` class that supports `mlflow`. Should be initialized at the start of
    your script.

    # WHAT: Tracker for MLflow experiment tracking
    # WHY: MLflow is popular for production ML workflows
    # HOW: Wraps mlflow.start_run() and mlflow.log_metrics()
    """

    name = "mlflow"
    requires_logging_directory = False  # Can use remote tracking server

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        logging_dir: Optional[Union[str, os.PathLike]] = None,
        run_id: Optional[str] = None,
        tags: Optional[Union[dict[str, Any], str]] = None,
        nested_run: Optional[bool] = False,
        run_name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        # WHAT: Read config from environment variables first
        # WHY: MLflow supports environment-based configuration
        # HOW: os.environ.get() with fallback to parameters

        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment_name)
        run_id = os.environ.get("MLFLOW_RUN_ID", run_id)
        tags = os.environ.get("MLFLOW_TAGS", tags)
        if isinstance(tags, str):
            tags = json.loads(tags)  # Parse JSON string

        nested_run = os.environ.get("MLFLOW_NESTED_RUN", nested_run)

        self.experiment_name = experiment_name
        self.logging_dir = logging_dir
        self.run_id = run_id
        self.tags = tags
        self.nested_run = nested_run
        self.run_name = run_name
        self.description = description

    @on_main_process
    def start(self):
        import mlflow

        # WHAT: Find or create MLflow experiment
        # WHY: MLflow uses experiments to group related runs
        # HOW: Search by name, create if not found

        exps = mlflow.search_experiments(filter_string=f"name = '{self.experiment_name}'")
        if len(exps) > 0:
            if len(exps) > 1:
                logger.warning("Multiple experiments with the same name found. Using first one.")
            experiment_id = exps[0].experiment_id
        else:
            experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                artifact_location=self.logging_dir,
                tags=self.tags,
            )

        # Start a new run within the experiment
        self.active_run = mlflow.start_run(
            run_id=self.run_id,
            experiment_id=experiment_id,
            run_name=self.run_name,
            nested=self.nested_run,
            tags=self.tags,
            description=self.description,
        )

    @property
    def tracker(self):
        return self.active_run

    @on_main_process
    def store_init_configuration(self, values: dict):
        import mlflow

        # WHAT: Log parameters with MLflow-specific validation
        # WHY: MLflow has strict limits on parameter length and batch size
        # HOW: Filter long values, batch into groups of 100

        for name, value in list(values.items()):
            # MLflow limit: 500 characters per parameter value
            if len(str(value)) > mlflow.utils.validation.MAX_PARAM_VAL_LENGTH:
                logger.warning_once(
                    f'Accelerate is attempting to log a value of "{value}" for key "{name}" as a parameter. '
                    f"MLflow's log_param() only accepts values no longer than {mlflow.utils.validation.MAX_PARAM_VAL_LENGTH} characters so we dropped this attribute."
                )
                del values[name]

        values_list = list(values.items())

        # MLflow limit: 100 parameters per batch
        for i in range(0, len(values_list), mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH):
            mlflow.log_params(dict(values_list[i : i + mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH]))

    @on_main_process
    def log(self, values: dict, step: Optional[int]):
        # WHAT: Log only numeric metrics
        # WHY: MLflow.log_metrics() only accepts int/float
        # HOW: Filter non-numeric values with warning

        metrics = {}
        for k, v in values.items():
            if isinstance(v, (int, float)):
                metrics[k] = v
            else:
                logger.warning_once(
                    f'MLflowTracker is attempting to log a value of "{v}" of type {type(v)} for key "{k}" as a metric. '
                    "MLflow's log_metric() only accepts float and int types so we dropped this attribute."
                )

        import mlflow
        mlflow.log_metrics(metrics, step=step)

    @on_main_process
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        # WHAT: Log entire directory as artifacts
        # WHY: MLflow stores models, plots, data as artifacts
        # HOW: mlflow.log_artifacts() uploads directory
        import mlflow
        mlflow.log_artifacts(local_dir=local_dir, artifact_path=artifact_path)

    @on_main_process
    def finish(self):
        import mlflow
        mlflow.end_run()
```

**Key Characteristics:**
- **Environment-aware**: Reads from `MLFLOW_*` environment variables
- **Validation**: Enforces MLflow's parameter length and batch size limits
- **Type filtering**: Only logs numeric metrics, warns on non-numeric
- **Artifact support**: Can log entire directories (models, plots, etc.)

---

## Common Patterns Across Trackers

### Pattern 1: Lazy Initialization

All trackers defer library imports and object creation to `start()`:

```python
def __init__(self, run_name: str, **kwargs):
    super().__init__()
    self.run_name = run_name
    self.init_kwargs = kwargs  # Store for later
    # NO import here!

def start(self):
    import wandb  # Import only when needed
    self.run = wandb.init(project=self.run_name, **self.init_kwargs)
```

**Why?** Avoids initializing `PartialState` too early, before distributed setup is complete.

###Pattern 2: Type Handling

Different trackers have different type requirements:

| Tracker | Scalars | Text | Dicts | Images | Tables |
|---------|---------|------|-------|--------|--------|
| TensorBoard | ✅ | ✅ | ✅ | ✅ | ❌ |
| WandB | ✅ | ✅ | ✅ | ✅ | ✅ |
| MLflow | ✅ | ❌ | ❌ | ✅ | ❌ |
| CometML | ✅ | ✅ | ✅ | ✅ | ❌ |
| Aim | ✅ | ✅ | ✅ | ✅ | ❌ |
| ClearML | ✅ | ❌ | ❌ | ✅ | ✅ |
| DVCLive | ✅ | ❌ | ❌ | ❌ | ❌ |
| SwanLab | ✅ | ✅ | ✅ | ✅ | ❌ |
| Trackio | ✅ | ✅ | ✅ | ❌ | ❌ |

### Pattern 3: Directory Requirements

**Requires Directory:**
- TensorBoard (stores event files)
- Aim (stores .aim repository)

**No Directory:**
- WandB (cloud-based)
- Comet ML (cloud-based)
- MLflow (can use remote tracking server)
- ClearML (cloud-based)
- DVCLive (flexible)
- SwanLab (cloud-based)
- Trackio (cloud-based)

### Pattern 4: Main Process Handling

**Main Process Only:**
- TensorBoard
- Comet ML
- Aim
- MLflow
- ClearML
- DVCLive

**All Processes (Internal Handling):**
- WandB (sets `main_process_only = False`)
- SwanLab (sets `main_process_only = False`)
- Trackio (sets `main_process_only = False`)

---

## Tracker Registry

**Location:** `src/accelerate/tracking.py:1258-1268`

```python
LOGGER_TYPE_TO_CLASS = {
    "aim": AimTracker,
    "comet_ml": CometMLTracker,
    "mlflow": MLflowTracker,
    "tensorboard": TensorBoardTracker,
    "wandb": WandBTracker,
    "clearml": ClearMLTracker,
    "dvclive": DVCLiveTracker,
    "swanlab": SwanLabTracker,
    "trackio": TrackioTracker,
}
```

**Purpose:** Maps tracker names (strings) to tracker classes for dynamic instantiation.

**Usage:**
```python
# In Accelerator.init_trackers()
tracker_class = LOGGER_TYPE_TO_CLASS["wandb"]
tracker_instance = tracker_class(run_name="my_experiment", ...)
```

---

## Helper Function: `filter_trackers()`

**Location:** `src/accelerate/tracking.py:1271-1327`

```python
def filter_trackers(
    log_with: list[Union[str, LoggerType, GeneralTracker]],
    logging_dir: Optional[Union[str, os.PathLike]] = None,
):
    """
    # WHAT: Validates and filters tracker list
    # WHY: Ensures requested trackers are available and properly configured
    # HOW: Checks package availability, validates directory requirements

    Takes in a list of potential tracker types and checks that:
        - The tracker wanted is available in that environment
        - Filters out repeats of tracker types
        - If `all` is in `log_with`, will return all trackers in the environment
        - If a tracker requires a `logging_dir`, ensures that `logging_dir` is not `None`
    """
    loggers = []

    if log_with is not None:
        if not isinstance(log_with, (list, tuple)):
            log_with = [log_with]

        if "all" in log_with or LoggerType.ALL in log_with:
            # Use all available trackers + any custom GeneralTracker instances
            loggers = [o for o in log_with if issubclass(type(o), GeneralTracker)] + get_available_trackers()
        else:
            for log_type in log_with:
                # Handle custom tracker instances
                if issubclass(type(log_type), GeneralTracker):
                    loggers.append(log_type)
                else:
                    log_type = LoggerType(log_type)
                    if log_type not in loggers:
                        if log_type in get_available_trackers():
                            tracker_init = LOGGER_TYPE_TO_CLASS[str(log_type)]
                            # Check if tracker requires logging directory
                            if tracker_init.requires_logging_directory:
                                if logging_dir is None:
                                    raise ValueError(
                                        f"Logging with `{log_type}` requires a `logging_dir` to be passed in."
                                    )
                            loggers.append(log_type)
                        else:
                            logger.debug(f"Tried adding logger {log_type}, but package is unavailable in the system.")

    return loggers
```

**Key Validation:**
1. Package availability check via `get_available_trackers()`
2. Directory requirement check for TensorBoard/Aim
3. Support for "all" to enable all available trackers
4. Support for custom `GeneralTracker` instances

---

## Creating a Custom Tracker

**Example: Discord Webhook Tracker**

```python
from accelerate.tracking import GeneralTracker, on_main_process
import requests

class DiscordTracker(GeneralTracker):
    """Send metrics to Discord webhook"""

    name = "discord"
    requires_logging_directory = False

    def __init__(self, run_name: str, webhook_url: str, **kwargs):
        super().__init__()
        self.run_name = run_name
        self.webhook_url = webhook_url

    @on_main_process
    def start(self):
        # Send start message
        requests.post(self.webhook_url, json={
            "content": f"🚀 Training started: {self.run_name}"
        })

    @property
    def tracker(self):
        return self.webhook_url

    @on_main_process
    def store_init_configuration(self, values: dict):
        # Send config as formatted message
        config_str = "\n".join(f"{k}: {v}" for k, v in values.items())
        requests.post(self.webhook_url, json={
            "content": f"**Config:**\n```\n{config_str}\n```"
        })

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        # Send metrics (only major milestones to avoid spam)
        if step and step % 100 == 0:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in values.items() if isinstance(v, float))
            requests.post(self.webhook_url, json={
                "content": f"Step {step}: {metrics_str}"
            })

    @on_main_process
    def finish(self):
        requests.post(self.webhook_url, json={
            "content": f"✅ Training completed: {self.run_name}"
        })

# Usage
discord_tracker = DiscordTracker("my_experiment", webhook_url="https://discord.com/api/webhooks/...")
accelerator = Accelerator(log_with=[discord_tracker, "tensorboard"])
accelerator.init_trackers("my_project")
```

---

## Summary

### Design Philosophy

1. **Unified Interface**: All trackers implement the same 4 methods (`start`, `store_init_configuration`, `log`, `finish`)
2. **Lazy Initialization**: Imports and setup deferred to `start()` to avoid distributed conflicts
3. **Flexible Requirements**: Some trackers need directories (TensorBoard), some don't (WandB)
4. **Distributed-Aware**: `@on_main_process` decorator prevents duplicate logging
5. **Type Handling**: Each tracker handles its supported data types differently
6. **Environment Integration**: MLflow reads from environment variables

### Integration Points

- **Accelerator.init_trackers()** → calls `tracker.start()` for each tracker
- **Accelerator.log()** → calls `tracker.log()` for each tracker
- **Accelerator.end_training()** → calls `tracker.finish()` for each tracker
- **accelerator.get_tracker(name)** → returns `tracker.tracker` for native API access

### File References

- **Base class**: `src/accelerate/tracking.py:101-180`
- **TensorBoard**: `src/accelerate/tracking.py:182-295`
- **WandB**: `src/accelerate/tracking.py:297-429`
- **MLflow**: `src/accelerate/tracking.py:705-910`
- **Registry**: `src/accelerate/tracking.py:1258-1268`
- **Filter function**: `src/accelerate/tracking.py:1271-1327`

### Next Steps

- **13_Experiment_Tracking_Tutorial.md** - Practical usage of `init_trackers()` and `log()`
- **14_Tracker_Feature_Comparison.md** - Detailed feature matrix
- **15_Multi_Tracker_Best_Practices.md** - Multi-tracker logging strategies
