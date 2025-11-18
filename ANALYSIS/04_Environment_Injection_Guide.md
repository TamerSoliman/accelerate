# Concept-to-Code: Distributed Environment Injection in Accelerate

## Overview
This guide explains **how** Accelerate transforms standard PyTorch code into distributed training code through **environment injection**. The core mechanism is the `prepare()` method, which wraps your models, optimizers, and dataloaders with distributed-aware versions.

---

## The Three Layers of Abstraction

### Layer 1: User Code (PyTorch Standard)
```python
# Standard PyTorch - runs on single GPU
model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
dataloader = DataLoader(dataset, batch_size=32)

for batch in dataloader:
    outputs = model(batch)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Layer 2: Accelerate Wrapper (Distributed-Aware)
```python
# Same code, but works on single GPU, multi-GPU, TPU, etc.
accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    outputs = model(batch)
    loss = loss_fn(outputs, labels)
    accelerator.backward(loss)  # Handles mixed precision + distributed
    optimizer.step()
    optimizer.zero_grad()
```

### Layer 3: What Actually Happens (Internal Wrappers)
```python
# What your objects become after prepare()
model = DistributedDataParallel(model)  # DDP wrapper
optimizer = AcceleratedOptimizer(optimizer, scaler=GradScaler())  # Scaler wrapper
dataloader = DataLoaderDispatcher(dataloader, DistributedSampler(...))  # Distributed sampler

# When you call model(batch):
# 1. Data is split across GPUs
# 2. Forward pass runs in parallel
# 3. Gradients are all-reduced automatically
# 4. Parameters are synchronized
```

---

## Role of prepare(): The Central Injection Point

### High-Level Flow
```
User calls: accelerator.prepare(model, optimizer, dataloader)
    ↓
Accelerator determines distributed backend (DDP/FSDP/DeepSpeed)
    ↓
Routes to backend-specific preparation method
    ↓
Each object is wrapped with distributed-aware version
    ↓
Returns wrapped objects (same interface, different behavior)
```

### Backend Routing Decision Tree
```python
# Inside prepare() - src/accelerate/accelerator.py:1413

if self.distributed_type == DistributedType.DEEPSPEED:
    result = self._prepare_deepspeed(*args)
    # Wraps with DeepSpeedEngine, DeepSpeedOptimizer

elif self.distributed_type == DistributedType.MEGATRON_LM:
    result = self._prepare_megatron_lm(*args)
    # Wraps with MegatronEngine, MegatronLMOptimizer

elif self.is_fsdp2:
    result = self._prepare_fsdp2(*args)
    # Wraps with FullyShardedDataParallel (PyTorch 2.6+)

else:
    # Standard path: DDP, FSDP1, Single GPU
    result = tuple(
        self._prepare_one(obj, first_pass=True, device_placement=d)
        for obj, d in zip(args, device_placement)
    )
    result = tuple(
        self._prepare_one(obj, device_placement=d)
        for obj, d in zip(result, device_placement)
    )
```

---

## Deep Dive: DDP (DistributedDataParallel) Injection

**Use Case:** Multi-GPU training on a single node or multiple nodes

### 1. Environment Detection (Before prepare())
**File:** `src/accelerate/state.py:PartialState.__init__`

```python
# Simplified version of how AcceleratorState detects DDP
class PartialState:
    def __init__(self):
        # Check if running in distributed environment
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            # Multi-process environment detected
            if not torch.distributed.is_initialized():
                # Initialize process group
                torch.distributed.init_process_group(
                    backend="nccl",  # For CUDA GPUs
                    init_method="env://",  # Read MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
                )
            self.distributed_type = DistributedType.MULTI_GPU
            self.num_processes = torch.distributed.get_world_size()
            self.process_index = torch.distributed.get_rank()
            self.local_process_index = int(os.environ.get("LOCAL_RANK", 0))
        else:
            # Single GPU or CPU
            self.distributed_type = DistributedType.NO
            self.num_processes = 1
            self.process_index = 0
            self.local_process_index = 0

        # Set device
        if self.distributed_type == DistributedType.MULTI_GPU:
            self.device = torch.device("cuda", self.local_process_index)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Key Environment Variables:**
- `WORLD_SIZE`: Total number of processes (e.g., 4 for 4 GPUs)
- `RANK`: Global rank of current process (0-3 for 4 GPUs)
- `LOCAL_RANK`: Local rank on current node (0-3 if all GPUs on same node)
- `MASTER_ADDR`: IP address of rank 0 process
- `MASTER_PORT`: Port for communication

**When is this set?**
- By `accelerate launch` command
- By `torchrun` or `torch.distributed.launch`
- Manually in your script

### 2. Model Wrapping with DDP
**File:** `src/accelerate/accelerator.py:1707` (prepare_model method)

```python
def prepare_model(self, model, device_placement=None, evaluation_mode=False):
    # ========================================================================
    # STEP 1: ADD MODEL TO TRACKING LIST
    # ========================================================================
    self._models.append(model)

    # ========================================================================
    # STEP 2: APPLY MIXED PRECISION (Wrap forward pass with autocast)
    # ========================================================================
    if self.native_amp:
        # Save original forward method
        model._original_forward = model.forward

        # Create autocast context manager
        autocast_context = get_mixed_precision_context_manager(
            self.native_amp, self.autocast_handler
        )

        # Wrap forward method with autocast
        # This ensures all forward pass ops use FP16/BF16
        model_forward_func = model.forward
        model.forward = convert_outputs_to_fp32(autocast_context(model_forward_func))

    # ========================================================================
    # STEP 3: MOVE MODEL TO DEVICE
    # ========================================================================
    if device_placement and not self.verify_device_map(model):
        model = model.to(self.device)
        # For DDP with 4 GPUs, this moves model to cuda:0, cuda:1, cuda:2, cuda:3

    # ========================================================================
    # STEP 4: WRAP WITH DDP (The Key Injection Point)
    # ========================================================================
    if not evaluation_mode:
        if self.multi_device:  # True for MULTI_GPU distributed_type
            if any(p.requires_grad for p in model.parameters()):
                # Get DDP configuration from handler
                kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}

                # **CRITICAL WRAPPING:** Create DistributedDataParallel wrapper
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.local_process_index],  # [0], [1], [2], [3] for 4 GPUs
                    output_device=self.local_process_index,  # 0, 1, 2, 3
                    **kwargs  # Additional DDP args (find_unused_parameters, etc.)
                )

                # Register communication hooks if specified
                if self.ddp_handler is not None:
                    self.ddp_handler.register_comm_hook(model)

    return model
```

**What DistributedDataParallel Does:**
1. **Initialization:**
   - Broadcasts parameters from rank 0 to all other ranks (ensures identical starting weights)
   - Registers backward hooks on all parameters
2. **Forward Pass:**
   - Each GPU processes a different slice of the batch
   - No communication during forward pass
3. **Backward Pass:**
   - Gradients are computed locally on each GPU
   - **Automatic gradient all-reduce** using `torch.distributed.all_reduce()`
   - All GPUs end up with averaged gradients
4. **Optimizer Step:**
   - Each GPU updates its local copy of the model
   - Since gradients are synchronized, all copies stay identical

### 3. DataLoader Wrapping
**File:** `src/accelerate/data_loader.py:prepare_data_loader`

```python
def prepare_data_loader(
    dataloader,
    device,
    num_processes=1,
    process_index=0,
    split_batches=False,
    put_on_device=False,
    rng_types=None,
    dispatch_batches=None,
    even_batches=True,
    use_seedable_sampler=False,
):
    # ========================================================================
    # STEP 1: EXTRACT ORIGINAL DATALOADER COMPONENTS
    # ========================================================================
    sampler = getattr(dataloader, "sampler", None)
    batch_sampler = getattr(dataloader, "batch_sampler", None)

    # ========================================================================
    # STEP 2: CREATE DISTRIBUTED SAMPLER
    # ========================================================================
    # **WHAT:** Replace standard sampler with DistributedSampler
    # **HOW:** Wrap existing sampler to distribute data across GPUs
    # **WHY:** Ensures each GPU sees different data, no overlap

    if num_processes > 1:
        # Create DistributedSampler that shards data
        if batch_sampler is None:
            # Standard sampler path
            new_sampler = DistributedSampler(
                dataloader.dataset,
                num_replicas=num_processes,  # Total number of GPUs
                rank=process_index,          # Current GPU rank (0-3)
                shuffle=isinstance(sampler, torch.utils.data.RandomSampler),
                seed=0,                      # For reproducibility
            )
        else:
            # Batch sampler path
            new_batch_sampler = DistributedSamplerWithLoop(
                batch_sampler,
                num_replicas=num_processes,
                rank=process_index,
            )

    # ========================================================================
    # STEP 3: RECONSTRUCT DATALOADER WITH NEW SAMPLER
    # ========================================================================
    # **WHAT:** Create new DataLoader with distributed sampler
    # **HOW:** Clone original DataLoader, replace sampler
    # **WHY:** Preserve all other DataLoader settings (num_workers, etc.)

    new_dataloader = DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=new_sampler,  # <-- Distributed sampler injected here
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        # ... other settings copied from original
    )

    # ========================================================================
    # STEP 4: WRAP IN DATALOADERDISPATCHER
    # ========================================================================
    # **WHAT:** Add additional functionality (device placement, state tracking)
    # **HOW:** Wrap DataLoader in custom iterator class
    # **WHY:** Automatically move batches to device, handle gradient accumulation

    if put_on_device or dispatch_batches:
        new_dataloader = DataLoaderDispatcher(
            new_dataloader,
            device=device,
            split_batches=split_batches,
            dispatch_batches=dispatch_batches,
        )

    return new_dataloader
```

**What DistributedSampler Does:**
```python
# Example with 4 GPUs and 100 samples
# Original indices: [0, 1, 2, ..., 99]

# GPU 0 (rank 0): [0, 4, 8, 12, ..., 96]
# GPU 1 (rank 1): [1, 5, 9, 13, ..., 97]
# GPU 2 (rank 2): [2, 6, 10, 14, ..., 98]
# GPU 3 (rank 3): [3, 7, 11, 15, ..., 99]

# Each GPU processes every 4th sample
# No overlap, full coverage of dataset
```

### 4. Optimizer Wrapping
**File:** `src/accelerate/optimizer.py:AcceleratedOptimizer`

```python
class AcceleratedOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, device_placement=True, scaler=None):
        self.optimizer = optimizer  # Original optimizer (e.g., AdamW)
        self.scaler = scaler         # GradScaler for FP16 (if mixed_precision='fp16')
        self.accelerator_state = AcceleratorState()
        self.gradient_state = GradientState()

        # Move optimizer state to device (for multi-GPU)
        if device_placement:
            state_dict = self.optimizer.state_dict()
            state_dict = move_to_device(state_dict, self.accelerator_state.device)
            self.optimizer.load_state_dict(state_dict)

    def step(self, closure=None):
        # ====================================================================
        # GRADIENT ACCUMULATION GATING
        # ====================================================================
        # **WHAT:** Only step when sync_gradients is True
        # **HOW:** Check global gradient_state flag
        # **WHY:** For gradient accumulation, we skip optimizer.step() until
        #          we've accumulated N gradient steps

        if self.gradient_state.sync_gradients:
            if self.scaler is not None:
                # FP16 mixed precision path
                # scaler.step() handles:
                # 1. Unscaling gradients
                # 2. Checking for inf/NaN
                # 3. Calling optimizer.step() if no overflow
                # 4. Updating scale factor
                self.scaler.step(self.optimizer, closure)
                self.scaler.update()
            else:
                # Standard path (BF16, FP32, or no mixed precision)
                self.optimizer.step(closure)

    def zero_grad(self, set_to_none=None):
        # Only zero gradients when sync_gradients is True
        # (respects gradient accumulation)
        if self.gradient_state.sync_gradients:
            self.optimizer.zero_grad(set_to_none=set_to_none)
```

**Why Wrap Optimizer?**
1. **Gradient Accumulation:** Skip updates until N batches processed
2. **Mixed Precision:** Handle GradScaler integration
3. **State Management:** Ensure optimizer state is on correct device

---

## Code Path Mapping: DDP Backend

### File Locations for DDP Flow

| **Concept** | **File Path** | **Function/Class** | **What It Does** |
|-------------|---------------|-------------------|------------------|
| **Environment Detection** | `src/accelerate/state.py:124` | `PartialState.__init__` | Detects MULTI_GPU via env vars, initializes torch.distributed |
| **Process Group Init** | `src/accelerate/state.py:250+` | `torch.distributed.init_process_group()` | Creates NCCL backend for GPU communication |
| **prepare() Entry** | `src/accelerate/accelerator.py:1413` | `Accelerator.prepare()` | Routes to _prepare_one for DDP |
| **Model Wrapping** | `src/accelerate/accelerator.py:1707` | `Accelerator.prepare_model()` | Wraps model with DistributedDataParallel |
| **DDP Wrapper** | PyTorch built-in | `torch.nn.parallel.DistributedDataParallel` | Handles gradient all-reduce |
| **Optimizer Wrapping** | `src/accelerate/optimizer.py:38` | `AcceleratedOptimizer` | Handles gradient accumulation + scaling |
| **DataLoader Wrapping** | `src/accelerate/data_loader.py:86` | `prepare_data_loader()` | Injects DistributedSampler |
| **Distributed Sampler** | PyTorch built-in | `torch.utils.data.DistributedSampler` | Shards data across GPUs |

### Critical Interception Points

#### 1. Model Forward Pass
```python
# User code:
outputs = model(batch)

# What actually happens (after prepare):
# 1. Batch is already sharded by DistributedSampler
# 2. model.forward() is wrapped with autocast context (if mixed_precision)
# 3. Each GPU computes forward pass on its batch slice independently
# 4. No inter-GPU communication during forward
```

**Injection Point:** `src/accelerate/accelerator.py:1750-1761`
```python
# Wraps model.forward with autocast
if self.native_amp:
    model._original_forward = model.forward
    autocast_context = get_mixed_precision_context_manager(self.native_amp, self.autocast_handler)
    model.forward = convert_outputs_to_fp32(autocast_context(model_forward_func))
```

#### 2. Loss Backward Pass
```python
# User code:
accelerator.backward(loss)

# What actually happens:
# 1. Loss is divided by gradient_accumulation_steps
# 2. If FP16: loss is scaled by scaler.scale()
# 3. loss.backward() computes gradients
# 4. DDP's registered hooks trigger gradient all-reduce across GPUs
# 5. All GPUs end up with identical averaged gradients
```

**Injection Point:** `src/accelerate/accelerator.py:2708-2740`
```python
def backward(self, loss, **kwargs):
    if self.distributed_type != DistributedType.DEEPSPEED:
        loss = loss / self.gradient_accumulation_steps
    if self.scaler is not None:
        self.scaler.scale(loss).backward(**kwargs)  # <-- Hooks trigger here
    else:
        loss.backward(**kwargs)
```

**DDP Hook Mechanism:**
- DDP registers a backward hook on every parameter
- When `loss.backward()` computes gradients:
  1. Hook detects gradient computation
  2. Triggers `torch.distributed.all_reduce(grad, op=SUM)`
  3. Gradients are averaged across all GPUs
  4. Result stored back in `param.grad`

#### 3. Optimizer Step
```python
# User code:
optimizer.step()

# What actually happens:
# 1. Check if gradients should be synchronized (gradient accumulation)
# 2. If FP16: scaler.step() checks for overflow, unscales, then steps
# 3. optimizer.step() updates parameters using averaged gradients
# 4. Each GPU updates its local copy independently
# 5. Since gradients are identical, all copies stay synchronized
```

**Injection Point:** `src/accelerate/optimizer.py:145-200`
```python
def step(self, closure=None):
    if self.gradient_state.sync_gradients:
        if self.scaler is not None:
            self.scaler.step(self.optimizer, closure)
            self.scaler.update()
        else:
            self.optimizer.step(closure)
```

---

## FSDP (Fully Sharded Data Parallel) Injection

**Use Case:** Training models too large to fit on a single GPU

### Key Differences from DDP

| **Aspect** | **DDP** | **FSDP** |
|------------|---------|----------|
| **Parameter Replication** | Full copy on each GPU | Sharded across GPUs |
| **Memory Usage** | N × model_size | model_size (approximately) |
| **Communication** | Gradients all-reduced | Parameters gathered + gradients reduced |
| **Forward Pass** | No communication | All-gather parameters before each layer |
| **Backward Pass** | All-reduce gradients | All-gather parameters, reduce-scatter gradients |

### FSDP Wrapping Process
**File:** `src/accelerate/accelerator.py:1843-1900` (prepare_model for FSDP)

```python
if self.distributed_type == DistributedType.FSDP:
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

    # Get FSDP configuration from plugin
    auto_wrap_policy = self.state.fsdp_plugin.auto_wrap_policy
    mixed_precision_policy = self.state.fsdp_plugin.mixed_precision_policy

    # **CRITICAL WRAPPING:** Wrap model with FSDP
    model = FSDP(
        model,
        sharding_strategy=self.state.fsdp_plugin.sharding_strategy,  # FULL_SHARD, SHARD_GRAD_OP, etc.
        cpu_offload=self.state.fsdp_plugin.cpu_offload,
        auto_wrap_policy=auto_wrap_policy,  # Which layers to shard
        backward_prefetch=self.state.fsdp_plugin.backward_prefetch,
        mixed_precision=mixed_precision_policy,
        device_id=self.device.index,
        **kwargs
    )
```

**What FSDP Does Differently:**

1. **Parameter Sharding:**
   ```python
   # Original model: 1B parameters × 4 bytes = 4GB per GPU
   # FSDP with 4 GPUs: 1B parameters / 4 = 250M params × 4 bytes = 1GB per GPU

   # Each GPU stores 1/4 of the parameters
   # GPU 0: params[0:250M]
   # GPU 1: params[250M:500M]
   # GPU 2: params[500M:750M]
   # GPU 3: params[750M:1B]
   ```

2. **Forward Pass Communication:**
   ```python
   # For each FSDP-wrapped layer:
   # 1. All-gather parameters from all GPUs
   # 2. Compute forward pass with full parameters
   # 3. Discard non-local parameters (free memory)
   # 4. Pass activations to next layer
   ```

3. **Backward Pass Communication:**
   ```python
   # For each FSDP-wrapped layer (in reverse order):
   # 1. All-gather parameters (reconstruct full layer)
   # 2. Compute gradients
   # 3. Reduce-scatter gradients (each GPU keeps its shard's gradients)
   # 4. Free non-local parameters
   ```

### FSDP Code Paths

| **Operation** | **File Path** | **Function** |
|--------------|---------------|--------------|
| **FSDP Wrapping** | `src/accelerate/accelerator.py:1843` | `prepare_model()` FSDP branch |
| **FSDP Plugin Config** | `src/accelerate/utils/dataclasses.py` | `FullyShardedDataParallelPlugin` |
| **FSDP Utilities** | `src/accelerate/utils/fsdp_utils.py` | State dict handling, checkpointing |
| **PyTorch FSDP** | PyTorch built-in | `torch.distributed.fsdp.FullyShardedDataParallel` |

---

## DeepSpeed Injection

**Use Case:** Maximum memory efficiency, ZeRO optimizations, pipeline parallelism

### DeepSpeed Preparation Flow
**File:** `src/accelerate/accelerator.py:2064` (_prepare_deepspeed)

```python
def _prepare_deepspeed(self, *args):
    # Extract model, optimizer, scheduler from args
    model = None
    optimizer = None
    scheduler = None
    for obj in args:
        if isinstance(obj, torch.nn.Module):
            model = obj
        elif isinstance(obj, torch.optim.Optimizer):
            optimizer = obj
        elif isinstance(obj, LRScheduler):
            scheduler = obj

    # **CRITICAL CALL:** Initialize DeepSpeed engine
    # This replaces all separate wrapping with a single engine
    import deepspeed

    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=deepspeed_config,  # ZeRO stage, offloading, etc.
        dist_init_required=False,  # We already initialized torch.distributed
    )

    # Wrap in Accelerate's DeepSpeed wrappers
    engine = DeepSpeedEngineWrapper(engine)
    optimizer = DeepSpeedOptimizerWrapper(optimizer)
    scheduler = DeepSpeedSchedulerWrapper(scheduler)

    return (engine, optimizer, scheduler, ...)
```

**Key DeepSpeed Features:**
- **ZeRO-1:** Shard optimizer states
- **ZeRO-2:** Shard gradients + optimizer states
- **ZeRO-3:** Shard parameters + gradients + optimizer states
- **CPU Offloading:** Move optimizer states to CPU RAM
- **Pipeline Parallelism:** Split model across GPUs

**File Paths:**
| **Component** | **File** |
|--------------|----------|
| **DeepSpeed Preparation** | `src/accelerate/accelerator.py:2064` |
| **DeepSpeed Plugin** | `src/accelerate/utils/dataclasses.py` (DeepSpeedPlugin) |
| **DeepSpeed Utilities** | `src/accelerate/utils/deepspeed.py` |
| **Wrapper Classes** | `src/accelerate/utils/deepspeed.py` (DeepSpeedEngineWrapper, etc.) |

---

## Summary: The Injection Matrix

| **Component** | **DDP** | **FSDP** | **DeepSpeed** |
|--------------|---------|----------|---------------|
| **Model** | DistributedDataParallel | FullyShardedDataParallel | DeepSpeedEngine |
| **Optimizer** | AcceleratedOptimizer | AcceleratedOptimizer | DeepSpeedOptimizer |
| **DataLoader** | DistributedSampler | DistributedSampler | DistributedSampler |
| **Communication** | Gradient all-reduce | All-gather params + reduce-scatter grads | ZeRO-based sharding |
| **Memory** | N × model_size | ~model_size | Depends on ZeRO stage |
| **Primary File** | `accelerator.py:1823` | `accelerator.py:1843` | `accelerator.py:2064` |

**Key Insight:** All backends follow the same pattern:
1. Detect environment (PartialState)
2. Route to backend-specific preparation (_prepare_deepspeed, _prepare_fsdp2, etc.)
3. Wrap objects with backend-specific wrappers
4. Return wrapped objects with standard interface

The user's code remains unchanged - only the internal behavior differs!
