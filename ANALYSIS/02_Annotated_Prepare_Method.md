# Annotated: Accelerator.prepare() Method

## File Location
**Source:** `src/accelerate/accelerator.py:1413`

## Overview
The `prepare()` method is the **core transformation engine** of Accelerate. It takes standard PyTorch objects (models, optimizers, dataloaders, schedulers) and **wraps** them with distributed-aware versions. This is where the magic happens: after calling `prepare()`, your training loop can use standard PyTorch code, but all operations are transparently distributed.

---

## Full Annotated Code

```python
def prepare(self, *args, device_placement=None):
    """
    Prepare all objects passed in `args` for distributed training and mixed precision, then return them in the same
    order.

    Args:
        *args (list of objects):
            Any of the following type of objects:
            - `torch.utils.data.DataLoader`: PyTorch Dataloader
            - `torch.nn.Module`: PyTorch Module
            - `torch.optim.Optimizer`: PyTorch Optimizer
            - `torch.optim.lr_scheduler.LRScheduler`: PyTorch LR Scheduler

        device_placement (`list[bool]`, *optional*):
            Used to customize whether automatic device placement should be performed for each object passed.
    """

    # ============================================================================
    # SECTION 1: DEVICE PLACEMENT CONFIGURATION
    # ============================================================================
    # **WHAT:** Setup device placement flags for each argument
    # **HOW:** Create a list of None values or validate user-provided list
    # **WHY:** Allows customization of which objects are automatically moved to device

    if device_placement is None:
        device_placement = [None for _ in args]
    elif self.distributed_type in (DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM):
        raise ValueError("You can't customize device placements with DeepSpeed or Megatron-LM.")
    elif len(device_placement) != len(args):
        raise ValueError(
            f"`device_placement` should be a list with {len(args)} elements (the number of objects passed)."
        )

    # ============================================================================
    # SECTION 2: VALIDATION CHECKS
    # ============================================================================
    # **WHAT:** Validate that objects are compatible with the current distributed setup
    # **HOW:** Check for device_map incompatibility and DeepSpeed multi-model limitations
    # **WHY:** Prevent silent failures or incorrect behavior

    # Check for device_map incompatibility (models loaded with accelerate's device_map='auto')
    for obj in args:
        if (
            isinstance(obj, torch.nn.Module)
            and self.verify_device_map(obj)
            and self.distributed_type != DistributedType.NO
            and os.environ.get("ACCELERATE_BYPASS_DEVICE_MAP", "false") != "true"
        ):
            raise ValueError(
                "You can't train a model that has been loaded with `device_map='auto'` in any distributed mode."
                " Please rerun your script specifying `--num_processes=1` or by launching with `python {{myscript.py}}`."
            )

    # DeepSpeed limitation: only one model per Accelerator instance
    if self.distributed_type == DistributedType.DEEPSPEED:
        model_count = 0
        for obj in args:
            if isinstance(obj, torch.nn.Module):
                model_count += 1
        if model_count > 1:
            raise AssertionError(
                "You can't use same `Accelerator()` instance with multiple models when using DeepSpeed"
            )

    # TPU/XLA specific check: model and optimizer must be on same device
    if self.distributed_type == DistributedType.XLA:
        model_device, optimizer_device = self._get_devices()
        if model_device is not None and optimizer_device is not None and model_device != optimizer_device:
            raise ValueError(
                "The model and the optimizer parameters are not on the same device, which probably means you "
                "created an optimizer around your model **before** putting on the device. Make sure the line "
                "model.to(device) is before the optimizer creation in your script or remove it entirely and use "
                "the flag default value for `device_placement` in your `Accelerator` to let it handle that "
                "part for you."
            )

    # FSDP2 specific validation: model and optimizer must be prepared together
    if self.is_fsdp2:
        model_count = 0
        optimizer_count = 0
        for i, obj in enumerate(args):
            if isinstance(obj, torch.nn.Module):
                model_count += 1
            elif isinstance(obj, torch.optim.Optimizer):
                optimizer_count += 1

        # Both model and optimizer must be present for FSDP2
        if (model_count < 1 and optimizer_count > 0) or (model_count > 0 and optimizer_count < 1):
            raise ValueError(
                "When using FSDP2, a model and optimizer must be passed together to `Accelerator.prepare()`"
                " as the optimizer needs to have its parameters modified after the model is converted."
            )
        if model_count > 1:
            raise ValueError("Only one model is supported when using FSDP2")

    # ============================================================================
    # SECTION 3: TPU PARAMETER TRACKING (Pre-wrapping)
    # ============================================================================
    # **WHAT:** Save a mapping of old model parameters before device placement
    # **HOW:** Use _get_named_parameters() to create a dict of parameter names to tensors
    # **WHY:** TPU/XLA creates NEW parameters when moving to device, so we need to update
    #          the optimizer's parameter references after wrapping

    tpu_should_fix_optimizer = self.device_placement and self.distributed_type == DistributedType.XLA

    if tpu_should_fix_optimizer:
        # Grab old model parameters (before .to(device) is called)
        old_named_params = self._get_named_parameters(*args, drop_refs=False)

    # ============================================================================
    # SECTION 4: IPEX OPTIMIZATION (Intel Extension for PyTorch)
    # ============================================================================
    # **WHAT:** Apply Intel optimizations for CPU/XPU training
    # **HOW:** Call self._prepare_ipex(*args) which uses torch_ipex.optimize()
    # **WHY:** Enables performance optimizations for Intel hardware

    if self.distributed_type in [DistributedType.MULTI_CPU, DistributedType.MULTI_XPU, DistributedType.NO]:
        if (
            is_torch_version("<", "2.7.0")
            and (self.device.type == "cpu" or self.device.type == "xpu")
            and self.state.use_ipex
        ):
            logger.warning(
                "You are using lower version of PyTorch(< 2.7.0) with ipex acceleration on Intel CPU or XPU, "
                "Intel has upstreamed most of the optimizations into stock PyTorch from 2.7.0, we encourage you "
                "to install the latest stock PyTorch and enjoy the out-of-experience on Intel CPU/XPU."
            )
            args = self._prepare_ipex(*args)

    # ============================================================================
    # SECTION 5: PARALLELISM CONFIGURATIONS (TP & CP)
    # ============================================================================
    # **WHAT:** Apply Tensor Parallelism (TP) or Context Parallelism (CP) if configured
    # **HOW:** Call specialized preparation methods (_prepare_tp, _prepare_cp)
    # **WHY:** Enables model parallelism across GPUs/devices

    if self.parallelism_config and self.parallelism_config.tp_enabled:
        args = self._prepare_tp(*args)

    if self.parallelism_config and self.parallelism_config.cp_enabled:
        args = self._prepare_cp(*args)

    # ============================================================================
    # SECTION 6: FP8 PREPARATION (Transformer Engine or TorchAO)
    # ============================================================================
    # **WHAT:** Convert models to FP8 format if fp8 mixed precision is enabled
    # **HOW:** Call backend-specific preparation (_prepare_te, _prepare_ao)
    # **WHY:** FP8 requires special model transformations before distributed wrapping

    if self.fp8_backend == FP8BackendType.TE:
        # Transformer Engine FP8
        args = self._prepare_te(*args)
    elif self.fp8_backend == FP8BackendType.AO:
        # TorchAO FP8
        args = self._prepare_ao(*args)

    # ============================================================================
    # SECTION 7: BACKEND-SPECIFIC PREPARATION (The Core Wrapping Logic)
    # ============================================================================
    # **WHAT:** Wrap objects based on the distributed backend (DeepSpeed, FSDP, DDP, etc.)
    # **HOW:** Route to backend-specific preparation methods
    # **WHY:** Each backend (DeepSpeed, FSDP, DDP) has different wrapping requirements

    if self.distributed_type == DistributedType.DEEPSPEED:
        # --- DeepSpeed Backend ---
        # **KEY FILE:** src/accelerate/accelerator.py:2064 (_prepare_deepspeed)
        # **WRAPS:**
        #   - Model → DeepSpeedEngine (via deepspeed.initialize())
        #   - Optimizer → DeepSpeedOptimizerWrapper
        #   - Scheduler → DeepSpeedSchedulerWrapper
        #   - DataLoader → Prepared DataLoader
        result = self._prepare_deepspeed(*args)

    elif self.distributed_type == DistributedType.MEGATRON_LM:
        # --- Megatron-LM Backend ---
        # **KEY FILE:** Uses megatron_lm_prepare_model_optimizer_scheduler
        # **WRAPS:**
        #   - Model → MegatronEngine
        #   - Optimizer → MegatronLMOptimizerWrapper
        result = self._prepare_megatron_lm(*args)

    elif self.is_fsdp2:
        # --- FSDP2 Backend (PyTorch 2.6+) ---
        # **KEY FILE:** src/accelerate/accelerator.py:1633 (_prepare_fsdp2)
        # **WRAPS:**
        #   - Model → torch.distributed.fsdp.FullyShardedDataParallel wrapper
        #   - Optimizer → Parameters switched to FSDP-wrapped parameters
        result = self._prepare_fsdp2(*args)

    else:
        # --- Standard Backends (DDP, FSDP1, Single GPU, etc.) ---
        # **KEY METHOD:** _prepare_one() called in two passes

        if self.fp8_backend == FP8BackendType.MSAMP:
            # MS-AMP requires special preparation
            args, device_placement = self._prepare_msamp(*args, device_placement=device_placement)

        # **FIRST PASS:** Prepare models, optimizers, dataloaders
        # **HOW:** Calls _prepare_one(obj, first_pass=True) for each object
        # **WHY:** Optimizers need to be prepared BEFORE schedulers (schedulers reference optimizers)
        result = tuple(
            self._prepare_one(obj, first_pass=True, device_placement=d) for obj, d in zip(args, device_placement)
        )

        # **SECOND PASS:** Prepare schedulers
        # **HOW:** Calls _prepare_one(obj, first_pass=False) for each object
        # **WHY:** Schedulers need the fully-prepared optimizer to wrap correctly
        result = tuple(self._prepare_one(obj, device_placement=d) for obj, d in zip(result, device_placement))

    # ============================================================================
    # SECTION 8: TPU PARAMETER FIXUP (Post-wrapping)
    # ============================================================================
    # **WHAT:** Update optimizer's parameter references after TPU device placement
    # **HOW:** Create a mapping from old params to new params and call optimizer._switch_parameters()
    # **WHY:** TPU creates new parameter objects when moving to XLA device, so optimizer
    #          still references the old (CPU) parameters. This updates the references.

    if tpu_should_fix_optimizer:
        # Grab new model parameters (after .to(device))
        new_named_params = self._get_named_parameters(*result)
        # Build mapping: old_param → new_param
        mapping = {p: new_named_params[n] for n, p in old_named_params.items()}
        # Update optimizer parameters
        for obj in result:
            if isinstance(obj, torch.optim.Optimizer):
                obj._switch_parameters(mapping)

    # ============================================================================
    # SECTION 9: MARK OBJECTS AS PREPARED
    # ============================================================================
    # **WHAT:** Add a flag to indicate objects have been prepared
    # **HOW:** Set _is_accelerate_prepared = True on each object
    # **WHY:** Prevents double-preparation and allows Accelerate to track prepared objects

    for item in result:
        if any(
            item in container
            for container in (self._dataloaders, self._models, self._optimizers, self._schedulers)
        ):
            item._is_accelerate_prepared = True

    # Return single object or tuple
    return result if len(result) > 1 else result[0]
```

---

## The Two-Pass Preparation Strategy

### Why Two Passes?
The `_prepare_one()` method is called **twice** for each object:

```python
# PASS 1: Prepare models, optimizers, dataloaders
result = tuple(self._prepare_one(obj, first_pass=True, ...) for obj, d in zip(args, device_placement))

# PASS 2: Prepare schedulers
result = tuple(self._prepare_one(obj, device_placement=d) for obj, d in zip(result, device_placement))
```

**Reason:** Schedulers (LRScheduler) need to reference the **wrapped** optimizer, not the original one. By delaying scheduler preparation to the second pass, we ensure schedulers see the correct optimizer.

---

## _prepare_one() Method Breakdown

**Location:** `src/accelerate/accelerator.py:1396`

```python
def _prepare_one(self, obj, first_pass=False, device_placement=None):
    # First pass: Prepare DataLoader, Model, Optimizer
    if first_pass:
        if isinstance(obj, torch.utils.data.DataLoader):
            return self.prepare_data_loader(obj, device_placement=device_placement)
        elif isinstance(obj, torch.nn.Module):
            return self.prepare_model(obj, device_placement=device_placement)
        elif isinstance(obj, torch.optim.Optimizer):
            optimizer = self.prepare_optimizer(obj, device_placement=device_placement)
            return optimizer
    # Second pass: Prepare LR scheduler
    elif isinstance(obj, LRScheduler):
        scheduler = self.prepare_scheduler(obj)
        return scheduler
    # Return unmodified if no criteria matched
    return obj
```

### What Each prepare_X Method Does:

1. **prepare_model(model)** → Wraps model with DDP/FSDP, applies mixed precision, moves to device
   - **File:** `src/accelerate/accelerator.py` (search for `def prepare_model`)
   - **Key wrapping:** `torch.nn.parallel.DistributedDataParallel` for DDP

2. **prepare_optimizer(optimizer)** → Wraps optimizer to intercept `.step()` and `.zero_grad()`
   - **File:** `src/accelerate/optimizer.py:AcceleratedOptimizer`
   - **Key wrapping:** `AcceleratedOptimizer` wrapper that handles gradient scaling

3. **prepare_data_loader(dataloader)** → Wraps dataloader for distributed sampling
   - **File:** `src/accelerate/data_loader.py:prepare_data_loader`
   - **Key wrapping:** `DataLoaderDispatcher` or modified DataLoader with DistributedSampler

4. **prepare_scheduler(scheduler)** → Wraps scheduler to coordinate with wrapped optimizer
   - **File:** `src/accelerate/scheduler.py:AcceleratedScheduler`
   - **Key wrapping:** `AcceleratedScheduler` wrapper

---

## Backend-Specific Preparation Deep Dive

### 1. DDP (DistributedDataParallel) Path

**Triggered when:** `self.distributed_type == DistributedType.MULTI_GPU` (and not DeepSpeed/FSDP)

**Flow:**
```
prepare() → _prepare_one() → prepare_model()
                           → Wraps with torch.nn.parallel.DistributedDataParallel
```

**Critical Code Location:**
```python
# In prepare_model() (accelerator.py, around line 1700+)
if self.distributed_type == DistributedType.MULTI_GPU:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[self.local_process_index],
        output_device=self.local_process_index,
        **self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {},
    )
```

**What this achieves:**
- Gradients are automatically all-reduced across GPUs during `backward()`
- Each GPU processes a different shard of the batch
- Model parameters are synchronized at initialization

---

### 2. FSDP (Fully Sharded Data Parallel) Path

**Triggered when:** `self.is_fsdp2 == True` or FSDP plugin is active

**Flow:**
```
prepare() → _prepare_fsdp2() → fsdp2_prepare_model()
                            → torch.distributed.fsdp.FullyShardedDataParallel
```

**Critical Code Location:**
```python
# In _prepare_fsdp2() (accelerator.py:1633)
def _prepare_fsdp2(self, *args):
    # First pass: prepare non-model objects
    result = [
        self._prepare_one(obj, first_pass=True) if not isinstance(obj, torch.nn.Module) else obj for obj in args
    ]

    # Wrap models with FSDP
    result = [self._prepare_one(obj) if not isinstance(obj, torch.nn.Module) else obj for obj in result]

    # Apply FSDP wrapping
    for i, obj in enumerate(result):
        if isinstance(obj, torch.nn.Module):
            result[i] = fsdp2_prepare_model(
                obj,
                self.state.fsdp_plugin,
                self.device,
            )

    # Switch optimizer parameters to FSDP-wrapped params
    fsdp2_switch_optimizer_parameters(result)

    return tuple(result)
```

**Key file:** `src/accelerate/utils/fsdp_utils.py`

**What this achieves:**
- Model parameters are sharded across GPUs (each GPU holds only a subset)
- During forward pass, parameters are gathered from all GPUs
- During backward pass, gradients are reduced and parameters are re-sharded
- Dramatically reduces memory usage for large models

---

### 3. DeepSpeed Path

**Triggered when:** `self.distributed_type == DistributedType.DEEPSPEED`

**Flow:**
```
prepare() → _prepare_deepspeed() → deepspeed.initialize()
                                → Returns DeepSpeedEngine, DeepSpeedOptimizer
```

**Critical Code Location:**
```python
# In _prepare_deepspeed() (accelerator.py:2064)
def _prepare_deepspeed(self, *args):
    # ... validation and setup ...

    # Initialize DeepSpeed engine
    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=deepspeed_config,
        dist_init_required=False,  # We already initialized torch.distributed
    )

    # Wrap in Accelerate's DeepSpeed wrappers
    return DeepSpeedEngineWrapper(engine), DeepSpeedOptimizerWrapper(optimizer), ...
```

**Key file:** `src/accelerate/utils/deepspeed.py`

**What this achieves:**
- DeepSpeed takes full control of the training loop
- Optimizer, gradient accumulation, and checkpointing are all managed by DeepSpeed
- Enables ZeRO optimization stages (ZeRO-1, ZeRO-2, ZeRO-3) for memory efficiency
- Supports advanced features like CPU offloading, mixed precision, and pipeline parallelism

---

## Critical Insight: The Wrapping Chain

After `prepare()`, your objects look like this:

```
Original:     model
Prepared:     DistributedDataParallel(model)  [DDP]
         OR:  FullyShardedDataParallel(model) [FSDP]
         OR:  DeepSpeedEngine(model)          [DeepSpeed]

Original:     optimizer
Prepared:     AcceleratedOptimizer(optimizer)  [Standard]
         OR:  DeepSpeedOptimizerWrapper(...)   [DeepSpeed]

Original:     dataloader
Prepared:     DataLoaderDispatcher(dataloader)
```

These wrappers **intercept** all method calls:
- `model.forward()` → routed through DDP/FSDP for distributed execution
- `optimizer.step()` → intercepted to handle gradient scaling, clipping, and synchronization
- `dataloader.__iter__()` → uses DistributedSampler to ensure each GPU gets different data

---

## Related Files

1. **Backend-specific preparation:**
   - DDP: Embedded in `src/accelerate/accelerator.py` (prepare_model method)
   - FSDP: `src/accelerate/utils/fsdp_utils.py` (fsdp2_prepare_model)
   - DeepSpeed: `src/accelerate/accelerator.py:2064` (_prepare_deepspeed)

2. **Wrapper classes:**
   - Optimizer: `src/accelerate/optimizer.py:AcceleratedOptimizer`
   - Scheduler: `src/accelerate/scheduler.py:AcceleratedScheduler`
   - DataLoader: `src/accelerate/data_loader.py:DataLoaderDispatcher`

3. **Helper functions:**
   - `prepare_data_loader`: `src/accelerate/data_loader.py:86`
   - `get_grad_scaler`: `src/accelerate/utils/__init__.py`
