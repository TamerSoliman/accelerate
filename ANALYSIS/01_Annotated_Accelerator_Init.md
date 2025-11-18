# Annotated: Accelerator.__init__() Method

## File Location
**Source:** `src/accelerate/accelerator.py:278`

## Overview
The `Accelerator.__init__()` method is the **entry point** for setting up the entire distributed training environment. It determines the execution context (CPU, single GPU, multi-GPU, TPU), initializes the appropriate backend (DDP, DeepSpeed, FSDP), and configures mixed precision training.

---

## Full Annotated Code

```python
def __init__(
    self,
    device_placement: bool = True,                    # Auto-move tensors to correct device
    split_batches: bool = _split_batches,            # Split batches across processes
    mixed_precision: PrecisionType | str | None = None,  # 'no', 'fp16', 'bf16', 'fp8'
    gradient_accumulation_steps: int = 1,            # Steps before gradient update
    cpu: bool = False,                               # Force CPU execution
    dataloader_config: DataLoaderConfiguration | None = None,
    deepspeed_plugin: DeepSpeedPlugin | dict[str, DeepSpeedPlugin] | None = None,
    fsdp_plugin: FullyShardedDataParallelPlugin | None = None,
    torch_tp_plugin: TorchTensorParallelPlugin | None = None,
    megatron_lm_plugin: MegatronLMPlugin | None = None,
    rng_types: list[str | RNGType] | None = None,
    log_with: str | LoggerType | GeneralTracker | list[str | LoggerType | GeneralTracker] | None = None,
    project_dir: str | os.PathLike | None = None,
    project_config: ProjectConfiguration | None = None,
    gradient_accumulation_plugin: GradientAccumulationPlugin | None = None,
    step_scheduler_with_optimizer: bool = True,
    kwargs_handlers: list[KwargsHandler] | None = None,
    dynamo_backend: DynamoBackend | str | None = None,
    dynamo_plugin: TorchDynamoPlugin | None = None,
    deepspeed_plugins: DeepSpeedPlugin | dict[str, DeepSpeedPlugin] | None = None,
    parallelism_config: ParallelismConfig | None = None,
):
    # ============================================================================
    # SECTION 1: PROJECT CONFIGURATION SETUP
    # ============================================================================
    # **WHAT:** Initialize project directories and logging paths
    # **HOW:** Uses ProjectConfiguration to manage output directories
    # **WHY:** Centralizes configuration for checkpoints, logs, and outputs

    self.trackers = []
    if project_config is not None:
        self.project_configuration = project_config
    else:
        self.project_configuration = ProjectConfiguration(project_dir=project_dir)
    if project_dir is not None and self.project_dir is None:
        self.project_configuration.set_directories(project_dir)

    # ============================================================================
    # SECTION 2: MIXED PRECISION VALIDATION
    # ============================================================================
    # **WHAT:** Validate the mixed_precision argument
    # **HOW:** Check against PrecisionType enum ('no', 'fp16', 'bf16', 'fp8')
    # **WHY:** Invalid precision modes would cause silent failures or crashes later

    if mixed_precision is not None:
        mixed_precision = str(mixed_precision)
        if mixed_precision not in PrecisionType:
            raise ValueError(
                f"Unknown mixed_precision mode: {mixed_precision}. Choose between {PrecisionType.list()}"
            )

    # ============================================================================
    # SECTION 3: BACKEND PLUGIN INITIALIZATION
    # ============================================================================
    # **WHAT:** Initialize DeepSpeed, FSDP, or other distributed training plugins
    # **HOW:** Reads from environment variables or passed plugin objects
    # **WHY:** Different backends (DeepSpeed, FSDP, DDP) require different initialization

    # --- DeepSpeed Plugin Setup ---
    if deepspeed_plugins is not None and deepspeed_plugin is not None:
        raise ValueError("You cannot pass in both `deepspeed_plugins` and `deepspeed_plugin`.")
    elif deepspeed_plugin is not None:
        deepspeed_plugins = deepspeed_plugin

    if deepspeed_plugins is None:
        # Check if another Accelerator instance already initialized DeepSpeed
        if (
            AcceleratorState._shared_state != {}
            and AcceleratorState().distributed_type == DistributedType.DEEPSPEED
        ):
            deepspeed_plugins = AcceleratorState().deepspeed_plugins
        else:
            # Initialize from environment variable ACCELERATE_USE_DEEPSPEED
            deepspeed_plugins = (
                DeepSpeedPlugin()
                if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
                else None
            )
    else:
        # Prevent creating a second Accelerator with different DeepSpeed config
        if (
            AcceleratorState._shared_state != {}
            and AcceleratorState().distributed_type == DistributedType.DEEPSPEED
            and AcceleratorState().deepspeed_plugins is not None
        ):
            raise NotImplementedError(
                "You cannot pass in a `deepspeed_plugin` when creating a second `Accelerator`. "
                "Please make sure the first `Accelerator` is initialized with all the plugins you want to use."
            )
        if isinstance(deepspeed_plugins, dict):
            for plugin in deepspeed_plugins.values():
                if not isinstance(plugin, DeepSpeedPlugin):
                    raise TypeError("`deepspeed_plugin` must be a DeepSpeedPlugin object.")

    # Set environment variable to enable DeepSpeed if plugin is provided
    if deepspeed_plugins is not None:
        os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
        if not is_deepspeed_available():
            raise ImportError("DeepSpeed is not installed => run `pip install deepspeed` or build it from source.")

        # Version checks for different hardware backends
        if is_mlu_available():
            if compare_versions("deepspeed", "<", "0.15.2"):
                raise ImportError("DeepSpeed MLU version must be >= 0.15.2. Please update DeepSpeed.")
        elif is_musa_available():
            if compare_versions("deepspeed", "<", "0.14.3"):
                raise ImportError("DeepSpeed MUSA version must be >= 0.14.3. Please update DeepSpeed.")
        elif compare_versions("deepspeed", "<", "0.9.3"):
            raise ImportError("DeepSpeed version must be >= 0.9.3. Please update DeepSpeed.")

        self.deepspeed_engine_wrapped = None

    # --- FSDP Plugin Setup ---
    if os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true" or isinstance(
        fsdp_plugin, FullyShardedDataParallelPlugin
    ):
        if not is_torch_version(">=", FSDP_PYTORCH_VERSION):
            raise ValueError(f"FSDP requires PyTorch >= {FSDP_PYTORCH_VERSION}")

    if fsdp_plugin is None:
        # Initialize from environment variable
        fsdp_plugin = (
            FullyShardedDataParallelPlugin()
            if os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
            else None
        )
    else:
        if not isinstance(fsdp_plugin, FullyShardedDataParallelPlugin):
            raise TypeError("`fsdp_plugin` must be a FullyShardedDataParallelPlugin object.")
        os.environ["ACCELERATE_USE_FSDP"] = "true"

    # FSDP2 requires PyTorch 2.6+
    if fsdp_plugin is not None and fsdp_plugin.fsdp_version == 2:
        if not is_torch_version(">=", FSDP2_PYTORCH_VERSION):
            raise ImportError(f"FSDP2 requires PyTorch >= {FSDP2_PYTORCH_VERSION}")

    # --- Megatron-LM Plugin Setup ---
    if megatron_lm_plugin is None:
        megatron_lm_plugin = (
            MegatronLMPlugin() if os.environ.get("ACCELERATE_USE_MEGATRON_LM", "false").lower() == "true" else None
        )
    else:
        if not isinstance(megatron_lm_plugin, MegatronLMPlugin):
            raise TypeError("`megatron_lm_plugin` must be a MegatronLMPlugin object.")
        os.environ["ACCELERATE_USE_MEGATRON_LM"] = "true"

    if megatron_lm_plugin:
        if not is_megatron_lm_available():
            raise ImportError("Megatron is not installed. please build it from source.")

    # ============================================================================
    # SECTION 4: KWARGS HANDLERS (Advanced Configuration)
    # ============================================================================
    # **WHAT:** Initialize handlers for advanced features (GradScaler, DDP, FP8, etc.)
    # **HOW:** Parse `kwargs_handlers` list and assign to specific handler attributes
    # **WHY:** Allows fine-grained control over backend-specific parameters

    self.ddp_handler = None           # DistributedDataParallelKwargs
    self.scaler_handler = None        # GradScalerKwargs
    self.init_handler = None          # InitProcessGroupKwargs
    self.fp8_recipe_handler = None    # FP8RecipeKwargs
    self.ao_recipe_handler = None     # AORecipeKwargs (TorchAO FP8)
    self.te_recipe_handler = None     # TERecipeKwargs (Transformer Engine FP8)
    self.msamp_recipe_handler = None  # MSAMPRecipeKwargs (MS-AMP FP8)
    self.autocast_handler = None      # AutocastKwargs
    self.profile_handler = None       # ProfileKwargs
    self.has_lomo_optimizer = False

    found_handlers = set()
    handler_class_to_attr = {
        DistributedDataParallelKwargs: "ddp_handler",
        GradScalerKwargs: "scaler_handler",
        InitProcessGroupKwargs: "init_handler",
        FP8RecipeKwargs: "fp8_recipe_handler",
        AutocastKwargs: "autocast_handler",
        ProfileKwargs: "profile_handler",
        AORecipeKwargs: "ao_recipe_handler",
        TERecipeKwargs: "te_recipe_handler",
        MSAMPRecipeKwargs: "msamp_recipe_handler",
    }
    self.has_fp8_handler = False
    if kwargs_handlers is not None:
        for handler in kwargs_handlers:
            assert isinstance(handler, KwargsHandler), (
                f"Unsupported kwargs handler passed: {handler}, must be one that inherits `accelerate.utils.KwargsHandler`."
            )
            # Ensure no duplicate handlers
            if handler.__class__ in found_handlers:
                raise ValueError(f"You can only pass one {handler.__class__} in `kwargs_handlers`.")
            found_handlers.add(handler.__class__)
            handler_attr = handler_class_to_attr[handler.__class__]
            setattr(self, handler_attr, handler)
            if "recipe_handler" in handler_attr and not self.has_fp8_handler:
                self.has_fp8_handler = True

    # ============================================================================
    # SECTION 5: ACCELERATORSTATE INITIALIZATION
    # ============================================================================
    # **WHAT:** Initialize the global state (singleton) that manages distributed setup
    # **HOW:** Calls AcceleratorState(), which detects environment and initializes torch.distributed
    # **WHY:** AcceleratorState determines the distributed backend (DDP, FSDP, DeepSpeed, etc.)
    #          and initializes the process group. This is the CORE of environment detection.

    kwargs = self.init_handler.to_kwargs() if self.init_handler is not None else {}
    self.state = AcceleratorState(
        mixed_precision=mixed_precision,
        cpu=cpu,
        dynamo_plugin=dynamo_plugin,
        deepspeed_plugin=deepspeed_plugins,
        fsdp_plugin=fsdp_plugin,
        megatron_lm_plugin=megatron_lm_plugin,
        parallelism_config=parallelism_config,
        _from_accelerator=True,
        **kwargs,
    )

    # **KEY INSIGHT:** After this call, self.state contains:
    #   - self.state.distributed_type (e.g., DistributedType.MULTI_GPU, DistributedType.DEEPSPEED)
    #   - self.state.device (the torch.device to use)
    #   - self.state.num_processes (total number of processes)
    #   - self.state.process_index (global rank of current process)
    #   - self.state.local_process_index (local rank on the node)

    # Setup parallelism device mesh if configured
    if self.parallelism_config:
        self.state.device_mesh = parallelism_config.get_device_mesh(self.device.type)
        self.parallelism_config._validate_accelerator(self)

    # ============================================================================
    # SECTION 6: MIXED PRECISION SETUP (GradScaler & Autocast)
    # ============================================================================
    # **WHAT:** Initialize GradScaler for FP16 training and determine if native AMP is used
    # **HOW:** Checks mixed_precision setting and device type, then creates GradScaler
    # **WHY:** FP16 requires loss scaling to prevent gradient underflow

    self.fp8_enabled = self.state.mixed_precision == "fp8" or mixed_precision == "fp8"

    # Auto-create FP8 recipe handlers if not provided
    if self.fp8_enabled and not self.has_fp8_handler:
        if self.fp8_backend == FP8BackendType.AO:
            self.ao_recipe_handler = AORecipeKwargs()
        elif self.fp8_backend == FP8BackendType.TE:
            self.te_recipe_handler = TERecipeKwargs()
        elif self.fp8_backend == FP8BackendType.MSAMP:
            self.msamp_recipe_handler = MSAMPRecipeKwargs()
        elif self.fp8_backend == FP8BackendType.NO:
            # Auto-detect available FP8 backend
            if is_torchao_available():
                logger.info("Found `torchao` installed, using it for FP8 training.")
                self.ao_recipe_handler = AORecipeKwargs()
            elif is_transformer_engine_available():
                logger.info("Found `transformer-engine` installed, using it for FP8 training.")
                self.te_recipe_handler = TERecipeKwargs()
            elif is_msamp_available():
                logger.info("Found `msamp` installed, using it for FP8 training.")
                self.msamp_recipe_handler = MSAMPRecipeKwargs()
            else:
                raise ImportError(
                    "Tried to train with `fp8` and auto-detect backend, but no FP8-compatible backend was installed. "
                    "Valid backends are: `torchao`, `transformer-engine`, and `msamp`."
                )
        self.has_fp8_handler = True

    # Determine if FP8 autocast should be delayed (for Transformer Engine)
    self.delayed_fp8_autocast = False
    if self.has_fp8_handler:
        if not self.fp8_enabled and (
            self.distributed_type not in (DistributedType.FSDP, DistributedType.DEEPSPEED)
        ):
            raise ValueError("Passing in an FP8 configuration requires setting `mixed_precision='fp8'`.")
        self.delayed_fp8_autocast = self.fp8_backend == "TE" and self.distributed_type in (
            DistributedType.MULTI_GPU,
            DistributedType.FSDP,
        )

    # Initialize gradient scaler and native AMP flag
    self.scaler = None
    self.native_amp = False

    # --- FP16 Mixed Precision ---
    if (
        self.state.mixed_precision == "fp16"
        and self.device.type != "cpu"
        and self.distributed_type not in (DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM)
    ):
        self.native_amp = True
        supported_device = ("xpu", "cuda", "npu", "xla", "mlu", "musa", "hpu", "sdaa", "mps")
        if self.device.type not in supported_device or is_torch_xla_available(check_is_tpu=True):
            raise ValueError(
                f"fp16 mixed precision requires a device in {supported_device} (not {self.device.type!r})."
            )
        if self.device.type == "mps" and not is_torch_version(">=", "2.5.0"):
            raise ValueError("fp16 mixed precision with MPS device requires a Pytorch >= 2.5.0")

        # **KEY STEP:** Create GradScaler for FP16
        kwargs = self.scaler_handler.to_kwargs() if self.scaler_handler is not None else {}

        # FSDP2 uses a different scaler
        if self.is_fsdp2:
            self.scaler = get_fsdp2_grad_scaler(device=self.device.type, **kwargs)
        else:
            self.scaler = get_grad_scaler(self.distributed_type, **kwargs)

    # --- BF16 Mixed Precision ---
    elif self.state.mixed_precision == "bf16" and self.distributed_type not in (
        DistributedType.DEEPSPEED,
        DistributedType.MEGATRON_LM,
    ):
        if self.device.type in ["cpu", "xpu", "hpu"]:
            self.native_amp = True
        else:
            self.native_amp = is_bf16_available(True)
        if not self.native_amp and not is_torch_xla_available():
            raise ValueError("bf16 mixed precision requires PyTorch >= 1.10 and a supported device.")
        if self.native_amp and self.device.type == "mps" and not is_torch_version(">=", "2.6.0"):
            raise ValueError("bf16 mixed precision with MPS device requires a Pytorch >= 2.6.0")

    # --- FP8 Mixed Precision ---
    elif self.fp8_enabled:
        self.native_amp = True
        if self.fp8_backend == FP8BackendType.MSAMP:
            if self.distributed_type == DistributedType.FSDP:
                raise NotImplementedError(
                    "`accelerate` + `MS-AMP` + `FSDP` is not supported at this time. "
                    "Please consider using deepspeed, which is supported."
                )
            elif self.distributed_type != DistributedType.DEEPSPEED:
                # MS-AMP requires GradScaler even with bf16 autocast
                self.scaler = get_grad_scaler(**kwargs)

    # ============================================================================
    # SECTION 7: INTERNAL STATE INITIALIZATION
    # ============================================================================
    # **WHAT:** Initialize tracking for models, optimizers, schedulers, dataloaders
    # **HOW:** Create empty lists to store references to prepared objects
    # **WHY:** Accelerator needs to track all prepared objects for state saving/loading

    self.step = 0
    self._optimizers = []
    self._models = []
    self._schedulers = []
    self._dataloaders = []
    self._custom_objects = []

    # Hooks for state loading/saving
    self._load_model_state_pre_hook = OrderedDict()
    self._save_model_state_pre_hook = OrderedDict()

    # RNG Types for reproducibility
    self.rng_types = rng_types
    if self.rng_types is None:
        self.rng_types = ["generator"]

    # Flag tensor for distributed early stopping
    self.flag_tensor = None

    # Final OS/kernel checks
    check_os_kernel()
```

---

## Key Takeaways

### 1. **Environment Detection Flow**
The initialization follows this critical path:
```
User calls Accelerator()
  → Plugins initialized (DeepSpeed/FSDP/Megatron)
  → AcceleratorState created (detects environment, initializes torch.distributed)
  → Mixed precision configured (GradScaler created if needed)
  → Internal state tracking initialized
```

### 2. **Three Critical Configuration Points**
1. **Backend Selection**: Lines 335-412 determine if using DeepSpeed, FSDP, or Megatron-LM
2. **AcceleratorState Initialization**: Lines 460-471 detect the distributed environment
3. **Mixed Precision Setup**: Lines 562-612 configure FP16/BF16/FP8 training

### 3. **The Role of AcceleratorState**
`AcceleratorState` (initialized at line 461) is a **singleton** that:
- Detects the current execution environment (CPU, single GPU, multi-GPU, TPU)
- Initializes `torch.distributed.init_process_group()` if needed
- Sets `self.device` to the appropriate `torch.device`
- Determines `self.distributed_type` (NO, MULTI_GPU, DEEPSPEED, FSDP, etc.)

**Location:** See `src/accelerate/state.py:124` for PartialState and AcceleratorState implementation.

### 4. **GradScaler Initialization Pattern**
For FP16 training (lines 577-583):
```python
if self.is_fsdp2:
    self.scaler = get_fsdp2_grad_scaler(device=self.device.type, **kwargs)
else:
    self.scaler = get_grad_scaler(self.distributed_type, **kwargs)
```

This creates a `torch.cuda.amp.GradScaler` (or `ShardedGradScaler` for FSDP) that will be used in:
- `accelerator.backward()` → `scaler.scale(loss).backward()`
- Optimizer step → `scaler.step(optimizer)` + `scaler.update()`

---

## Related Files
- **AcceleratorState**: `src/accelerate/state.py:124`
- **GradScaler utilities**: `src/accelerate/utils/__init__.py` (get_grad_scaler function)
- **Configuration dataclasses**: `src/accelerate/utils/dataclasses.py`
- **Backend plugins**:
  - DeepSpeed: `src/accelerate/utils/deepspeed.py`
  - FSDP: `src/accelerate/utils/fsdp_utils.py`
