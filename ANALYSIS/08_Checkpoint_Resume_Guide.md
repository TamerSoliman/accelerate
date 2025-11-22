# Enhancement 2: Checkpoint & Resume Deep Dive

## Overview
This guide provides a comprehensive understanding of Accelerate's checkpointing system, covering state saving/loading patterns, best practices, and backend-specific behaviors. Checkpointing is critical for:
- **Fault tolerance:** Resume training after crashes or preemption
- **Experimentation:** Save checkpoints at different epochs, compare models
- **Reproducibility:** Exact training continuation with RNG state restoration

---

## Architecture: What Gets Saved?

###checkpointed State Components

```
checkpoint_directory/
├── model.safetensors          # Model weights (safetensors format)
├── model.bin                   # Model weights (pickle format, fallback)
├── optimizer.bin               # Optimizer state (momentum, variance, etc.)
├── scheduler.bin               # LR scheduler state
├── sampler.bin                 # DataLoader sampler state (optional)
├── scaler.pt                   # GradScaler state (FP16 only)
├── random_states_0.pkl         # RNG states for process 0
├── random_states_1.pkl         # RNG states for process 1
├── ...                         # Additional files for multiple models/optimizers
└── custom_checkpoint_0.pkl     # Custom registered objects
```

**Total Checkpoint Size Example (GPT-2 Medium, 355M params):**
- Model: ~1.4 GB (FP32 weights)
- Optimizer (AdamW): ~2.8 GB (2× model size for momentum + variance)
- Scheduler: ~1 KB
- RNG states: ~10 KB
- **Total:** ~4.2 GB

---

## Core Functions

### 1. accelerator.save_state()

**Location:** `src/accelerate/accelerator.py:3467`

```python
def save_state(
    self,
    output_dir: str | None = None,
    safe_serialization: bool = True,
    **save_model_func_kwargs
):
    """
    Saves complete training state to a directory.

    Args:
        output_dir: Directory to save checkpoint
        safe_serialization: Use safetensors (True) or pickle (False)
        save_model_func_kwargs: Backend-specific save arguments

    What Gets Saved:
        1. Model weights (all models if multiple)
        2. Optimizer states (momentum, variance, etc.)
        3. LR scheduler states
        4. DataLoader sampler states
        5. GradScaler state (if FP16)
        6. RNG states (Python, NumPy, PyTorch, CUDA)
        7. Step counter
        8. Custom registered objects
    """
```

**Example Usage:**
```python
accelerator = Accelerator()
model, optimizer, dataloader, scheduler = accelerator.prepare(...)

# Train for some epochs
for epoch in range(10):
    train_one_epoch(model, dataloader, optimizer)

    # Save checkpoint every epoch
    checkpoint_dir = f"checkpoint_epoch_{epoch}"
    accelerator.save_state(checkpoint_dir)
```

---

### 2. accelerator.load_state()

**Location:** `src/accelerate/accelerator.py:3633`

```python
def load_state(
    self,
    input_dir: str | None = None,
    load_kwargs: dict | None = None,
    **load_model_func_kwargs
):
    """
    Loads complete training state from a directory.

    Args:
        input_dir: Directory containing checkpoint
        load_kwargs: Additional load arguments (e.g., map_location)
        load_model_func_kwargs: Model-specific load arguments

    What Gets Loaded:
        1. Model weights
        2. Optimizer states
        3. LR scheduler states
        4. DataLoader sampler states
        5. GradScaler state
        6. RNG states (restores exact random state)
        7. Step counter
        8. Custom registered objects

    Note: Must prepare() objects BEFORE loading state!
    """
```

**Example Usage:**
```python
accelerator = Accelerator()
model, optimizer, dataloader, scheduler = accelerator.prepare(...)

# Load checkpoint
accelerator.load_state("checkpoint_epoch_5")

# Continue training from epoch 6
for epoch in range(6, 10):
    train_one_epoch(model, dataloader, optimizer)
```

---

## Detailed Checkpoint Flow

### Save Flow (Step-by-Step)

```python
accelerator.save_state(output_dir)
    ↓
1. Create output directory
    os.makedirs(output_dir, exist_ok=True)

2. Handle automatic checkpoint naming (if enabled)
    - Create checkpoints/checkpoint_N/ structure
    - Delete old checkpoints if over total_limit

3. Wait for all processes (distributed sync)
    accelerator.wait_for_everyone()

4. Backend-specific model saving
    ├─ FSDP → save_fsdp_model()
    ├─ DeepSpeed → model.save_checkpoint()
    ├─ Megatron-LM → model.save_checkpoint()
    └─ Standard → extract state_dict with unwrapping

5. Save optimizers
    ├─ FSDP → save_fsdp_optimizer()
    ├─ DeepSpeed → (handled by DeepSpeed in step 4)
    └─ Standard → optimizer.state_dict()

6. Save schedulers
    scheduler.state_dict() → scheduler.bin

7. Save dataloader samplers
    sampler.state_dict() → sampler.bin (if SeedableRandomSampler)

8. Save GradScaler (if FP16)
    scaler.state_dict() → scaler.pt

9. Save RNG states (PER PROCESS!)
    random_states_{process_index}.pkl:
        - Python random state
        - NumPy RNG state
        - PyTorch CPU RNG state
        - PyTorch CUDA RNG states (all GPUs)
        - PyTorch XLA RNG state (if TPU)
        - Step counter

10. Call custom save hooks
    for hook in self._save_model_state_pre_hook.values():
        hook(models, weights, output_dir)

11. Save custom registered objects
    for obj in self._custom_objects:
        obj.state_dict() → custom_checkpoint_{i}.pkl
```

---

### Load Flow (Step-by-Step)

```python
accelerator.load_state(input_dir)
    ↓
1. Validate input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(...)

2. Call custom load hooks
    for hook in self._load_model_state_pre_hook.values():
        hook(models, input_dir)

3. Backend-specific model loading
    ├─ FSDP → load_fsdp_model()
    ├─ DeepSpeed → model.load_checkpoint()
    ├─ Megatron-LM → model.load_checkpoint()
    └─ Standard → model.load_state_dict(checkpoint)

4. Load optimizers
    ├─ FSDP → load_fsdp_optimizer()
    ├─ DeepSpeed → (handled by DeepSpeed in step 3)
    └─ Standard → optimizer.load_state_dict(checkpoint)

5. Load schedulers
    scheduler.load_state_dict(checkpoint)

6. Load dataloader samplers
    sampler.load_state_dict(checkpoint) (if exists)

7. Load GradScaler (if exists)
    scaler.load_state_dict(checkpoint)

8. Load RNG states
    Load random_states_{process_index}.pkl:
        random.setstate(states["random_state"])
        np.random.set_state(states["numpy_random_seed"])
        torch.set_rng_state(states["torch_manual_seed"])
        torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
        torch.xla.set_rng_state(states["xm_seed"])

9. Update step counter
    accelerator.step = states["step"]

10. Load custom registered objects
    for obj in self._custom_objects:
        obj.load_state_dict(checkpoint)
```

---

## RNG State Management (Reproducibility)

### What RNG States Are Saved?

**Location:** `src/accelerate/checkpointing.py:154-176`

```python
states = {}
states["step"] = step

# 1. Python's random module
states["random_state"] = random.getstate()

# 2. NumPy's RNG
states["numpy_random_seed"] = np.random.get_state()

# 3. PyTorch CPU RNG
states["torch_manual_seed"] = torch.get_rng_state()

# 4. PyTorch CUDA RNG (ALL GPUs!)
if is_cuda_available():
    states["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()

# 5. PyTorch XPU RNG (Intel GPUs)
if is_xpu_available():
    states["torch_xpu_manual_seed"] = torch.xpu.get_rng_state_all()

# 6. PyTorch MLU RNG (Cambricon MLU)
if is_mlu_available():
    states["torch_mlu_manual_seed"] = torch.mlu.get_rng_state_all()

# 7. PyTorch XLA RNG (TPU)
if is_torch_xla_available():
    states["xm_seed"] = xm.get_rng_state()
```

**Why This Matters:**
- **Data shuffling:** DataLoader uses Python's random module
- **Data augmentation:** Often uses NumPy or PyTorch RNG
- **Dropout:** Uses PyTorch's RNG
- **Weight initialization:** Uses PyTorch's RNG

**Without RNG state restoration:**
```python
# Training run 1:
set_seed(42)
train_epochs(1-5)
save_checkpoint("epoch_5")
continue_training(6-10)
# Result: Model accuracy at epoch 10 = 85.3%

# Training run 2 (resume from checkpoint):
set_seed(42)  # ❌ This resets RNG to initial state!
load_checkpoint("epoch_5")
continue_training(6-10)
# Result: Model accuracy at epoch 10 = 84.7% (DIFFERENT!)
```

**With RNG state restoration:**
```python
# Training run 2 (resume from checkpoint):
load_checkpoint("epoch_5")  # ✅ Restores RNG state from epoch 5
continue_training(6-10)
# Result: Model accuracy at epoch 10 = 85.3% (IDENTICAL!)
```

---

## Backend-Specific Behaviors

### 1. Standard DDP/Single GPU

**Save:**
```python
# Model: Extract state_dict, unwrap DDP wrapper
if isinstance(model, DistributedDataParallel):
    state_dict = model.module.state_dict()
else:
    state_dict = model.state_dict()

# Save with safetensors or pickle
save(state_dict, "model.safetensors", safe_serialization=True)
```

**Load:**
```python
# Load checkpoint
state_dict = load("model.safetensors")

# Load into model (DDP wrapper transparent)
model.load_state_dict(state_dict)
```

**Key Point:** DDP wrapper is transparent; only unwrapped model weights are saved.

---

### 2. FSDP (Fully Sharded Data Parallel)

**Save:**
```python
from accelerate.utils import save_fsdp_model

# FSDP-specific save (gathers sharded parameters)
save_fsdp_model(
    fsdp_plugin,
    accelerator,
    model,
    output_dir,
    model_index
)

# What happens internally:
# 1. Gather full parameters from all shards
# 2. Save on rank 0 only (or all ranks if save_on_each_node=True)
# 3. Save optimizer state separately
```

**Location:** `src/accelerate/utils/fsdp_utils.py:save_fsdp_model`

**Load:**
```python
from accelerate.utils import load_fsdp_model

# FSDP-specific load
load_fsdp_model(
    fsdp_plugin,
    accelerator,
    model,
    input_dir,
    model_index
)

# What happens internally:
# 1. Load full checkpoint
# 2. Distribute parameters across shards
# 3. Each GPU gets its assigned shard
```

**Critical Difference:** FSDP saves **full model** (gathered from shards), then re-shards on load.

**Optimizer State:**
```python
# FSDP optimizer state is also sharded!
save_fsdp_optimizer(fsdp_plugin, accelerator, optimizer, model, output_dir, index)

# Saves:
# - optimizer_0/: Shard for rank 0
# - optimizer_1/: Shard for rank 1
# - ...

# Load restores sharded optimizer state to each GPU
load_fsdp_optimizer(fsdp_plugin, accelerator, optimizer, model, input_dir, index)
```

---

### 3. DeepSpeed

**Save:**
```python
# DeepSpeed handles model + optimizer together
model.save_checkpoint(output_dir, ckpt_id="model")

# What happens internally:
# - DeepSpeed saves ZeRO-sharded states
# - Optimizer state saved based on ZeRO stage
# - Scheduler NOT saved by DeepSpeed (Accelerate handles separately)
```

**Directory Structure (DeepSpeed):**
```
checkpoint/
├── model/
│   ├── mp_rank_00_model_states.pt      # Model shard 0
│   ├── mp_rank_01_model_states.pt      # Model shard 1
│   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt  # Optimizer shard 0
│   └── zero_pp_rank_0_mp_rank_01_optim_states.pt  # Optimizer shard 1
└── scheduler.bin  # Saved by Accelerate
```

**Load:**
```python
# DeepSpeed handles loading
model.load_checkpoint(output_dir, ckpt_id="model")

# What happens internally:
# - Loads ZeRO-sharded model states
# - Loads ZeRO-sharded optimizer states
# - Reconstructs full training state
```

**Key Point:** DeepSpeed uses its own checkpoint format; not compatible with standard PyTorch checkpoints.

---

### 4. Megatron-LM

**Save/Load:**
```python
# Megatron-LM handles everything internally
model.save_checkpoint(output_dir)
model.load_checkpoint(output_dir)

# Saves model, optimizer, AND scheduler together
# Accelerate doesn't need to save scheduler separately
```

---

## Automatic Checkpoint Naming

### Configuration

```python
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

project_config = ProjectConfiguration(
    project_dir="./training_run",      # Root directory
    automatic_checkpoint_naming=True,  # Enable auto-naming
    total_limit=3,                      # Keep only 3 most recent checkpoints
    iteration=0,                        # Starting iteration (auto-incremented)
)

accelerator = Accelerator(project_config=project_config)
```

### Behavior

```python
# First save_state() call:
accelerator.save_state()
# Creates: training_run/checkpoints/checkpoint_0/

# Second save_state() call:
accelerator.save_state()
# Creates: training_run/checkpoints/checkpoint_1/

# Third save_state() call:
accelerator.save_state()
# Creates: training_run/checkpoints/checkpoint_2/

# Fourth save_state() call (total_limit=3):
accelerator.save_state()
# Creates: training_run/checkpoints/checkpoint_3/
# Deletes: training_run/checkpoints/checkpoint_0/ (oldest)

# Directory now contains: checkpoint_1, checkpoint_2, checkpoint_3
```

**Code Path:** `src/accelerate/accelerator.py:3505-3530`

```python
if self.project_configuration.automatic_checkpoint_naming:
    output_dir = os.path.join(self.project_dir, "checkpoints")
    # ...
    if total_limit is not None and (len(folders) + 1 > total_limit):
        # Sort by checkpoint number
        folders.sort(key=lambda f: int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", f)[0]))

        # Delete oldest checkpoints
        for folder in folders[:len(folders) + 1 - total_limit]:
            shutil.rmtree(folder)

    output_dir = os.path.join(output_dir, f"checkpoint_{self.save_iteration}")
    self.project_configuration.iteration += 1
```

---

## Custom Object Checkpointing

### Registering Custom Objects

```python
class MyCustomTracker:
    def __init__(self):
        self.loss_history = []
        self.metric_history = []

    def state_dict(self):
        return {
            "loss_history": self.loss_history,
            "metric_history": self.metric_history,
        }

    def load_state_dict(self, state_dict):
        self.loss_history = state_dict["loss_history"]
        self.metric_history = state_dict["metric_history"]

# Register with Accelerator
tracker = MyCustomTracker()
accelerator.register_for_checkpointing(tracker)

# Now tracker state is saved/loaded automatically
accelerator.save_state("checkpoint")
accelerator.load_state("checkpoint")
```

**Code Path:** `src/accelerate/accelerator.py:1232`

```python
def register_for_checkpointing(self, *objects):
    """
    Registers objects to be automatically checkpointed.

    Objects must have state_dict() and load_state_dict() methods.
    """
    self._custom_objects.extend(objects)
```

**Saved as:** `checkpoint/custom_checkpoint_{index}.pkl`

---

## Checkpoint Hooks

### Save Hooks

```python
def my_save_hook(models, weights, output_dir):
    """
    Custom hook called BEFORE model weights are saved.

    Args:
        models: List of model instances
        weights: List of model state_dicts
        output_dir: Directory where checkpoint will be saved

    Use cases:
        - Custom weight processing (pruning, quantization)
        - Logging checkpoint metadata
        - Validation before save
    """
    print(f"Saving checkpoint to {output_dir}")
    # Custom logic here

accelerator.register_save_state_pre_hook(my_save_hook)
```

### Load Hooks

```python
def my_load_hook(models, input_dir):
    """
    Custom hook called BEFORE model weights are loaded.

    Args:
        models: List of model instances
        input_dir: Directory containing checkpoint

    Use cases:
        - Pre-loading model modifications
        - Checkpoint validation
        - Custom weight mapping
    """
    print(f"Loading checkpoint from {input_dir}")
    # Custom logic here

accelerator.register_load_state_pre_hook(my_load_hook)
```

**Code Path:** `src/accelerate/accelerator.py:3602` (register hooks)

---

## Best Practices

### 1. Always Call wait_for_everyone() After save_state()

```python
# ❌ BAD: Ranks diverge after save
if accelerator.is_main_process:
    accelerator.save_state("checkpoint")
# Rank 1-3 continue training while rank 0 is saving!

# ✅ GOOD: All ranks wait
if accelerator.is_main_process:
    accelerator.save_state("checkpoint")
accelerator.wait_for_everyone()
```

**Why:** Rank 0 may take longer to save (especially with large models), causing other ranks to start next epoch early.

---

### 2. Use safe_serialization=True (Safetensors)

```python
# ✅ Recommended: Safetensors format
accelerator.save_state("checkpoint", safe_serialization=True)

# Benefits:
# - Faster save/load (no pickle overhead)
# - Safer (no arbitrary code execution)
# - Better error handling (detects corruption)

# ❌ Only use pickle if necessary (e.g., custom layers with state)
accelerator.save_state("checkpoint", safe_serialization=False)
```

---

### 3. Save Checkpoints to Fast Storage

```python
# ❌ BAD: Save to slow NFS/network storage
accelerator.save_state("/mnt/slow_nfs/checkpoint")
# Saving 4 GB checkpoint over network: 2-5 minutes

# ✅ GOOD: Save to local fast storage (NVMe SSD)
accelerator.save_state("/local/nvme/checkpoint")
# Saving 4 GB checkpoint to NVMe: 5-10 seconds

# Then asynchronously copy to network storage
if accelerator.is_main_process:
    subprocess.Popen(["rsync", "-a", "/local/nvme/checkpoint", "/mnt/nfs/backup/"])
```

---

### 4. Validate Checkpoints After Saving

```python
def validate_checkpoint(checkpoint_dir):
    """Verify checkpoint integrity after save"""
    required_files = ["model.safetensors", "optimizer.bin", "scheduler.bin"]
    for file in required_files:
        path = os.path.join(checkpoint_dir, file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing checkpoint file: {file}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"Checkpoint file is empty: {file}")

accelerator.save_state("checkpoint")
if accelerator.is_main_process:
    validate_checkpoint("checkpoint")
accelerator.wait_for_everyone()
```

---

### 5. Use total_limit to Prevent Disk Exhaustion

```python
# ❌ BAD: Save checkpoints indefinitely
for epoch in range(1000):
    train_epoch()
    accelerator.save_state(f"checkpoint_epoch_{epoch}")
# After 1000 epochs: 4.2 TB of checkpoints!

# ✅ GOOD: Automatic checkpoint rotation
project_config = ProjectConfiguration(
    project_dir="./training",
    automatic_checkpoint_naming=True,
    total_limit=5  # Keep only 5 most recent
)
accelerator = Accelerator(project_config=project_config)

for epoch in range(1000):
    train_epoch()
    accelerator.save_state()  # Auto-rotates, keeps only 5
# After 1000 epochs: Only 21 GB (5 checkpoints)
```

---

## Debugging Checkpoint Issues

### Common Issues & Solutions

#### 1. **"Checkpoint directory already exists"**
```python
# Error: ValueError: Checkpoint directory checkpoint_5 already exists.

# Cause: Tried to save to same iteration number twice

# Solution: Reset iteration counter
accelerator.project_configuration.iteration = 6
accelerator.save_state()
```

#### 2. **"Could not load random states"**
```python
# Warning: Could not load random states

# Cause: RNG states file missing (corrupted checkpoint)

# Impact: Training continues but won't be exactly reproducible

# Solution: Save checkpoints with verification
```

#### 3. **"Model and optimizer parameters don't match"**
```python
# Error: Size mismatch for parameter 'linear.weight'

# Cause: Loading checkpoint from different model architecture

# Solution: Validate model architecture before loading
def validate_architecture(model, checkpoint_dir):
    checkpoint = torch.load(os.path.join(checkpoint_dir, "model.bin"))
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint.keys())

    if model_keys != checkpoint_keys:
        missing = model_keys - checkpoint_keys
        unexpected = checkpoint_keys - model_keys
        raise ValueError(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")
```

#### 4. **FSDP Checkpoint Size Explosion**
```python
# Problem: FSDP checkpoint is 4× model size instead of ~1× model size

# Cause: save_on_each_node=False but each rank is saving

# Solution: Ensure only rank 0 saves (or all ranks if save_on_each_node=True)
project_config = ProjectConfiguration(
    save_on_each_node=False  # Only rank 0 saves
)
```

---

## Performance Optimization

### 1. Asynchronous Checkpointing

```python
import threading

def async_save_checkpoint(accelerator, output_dir):
    """Save checkpoint in background thread"""
    thread = threading.Thread(
        target=accelerator.save_state,
        args=(output_dir,),
        kwargs={"safe_serialization": True}
    )
    thread.start()
    return thread

# Save asynchronously
save_thread = async_save_checkpoint(accelerator, "checkpoint_epoch_10")

# Continue training (checkpoint saves in background)
train_epoch_11()

# Wait for checkpoint to finish before next save
save_thread.join()
```

**Caution:** Only safe if model/optimizer states don't change during save!

---

### 2. Checkpoint Sharding (Large Models)

```python
from huggingface_hub import split_torch_state_dict_into_shards

# Shard large checkpoints (e.g., 100 GB model)
state_dict = model.state_dict()

shards, index = split_torch_state_dict_into_shards(
    state_dict,
    max_shard_size="2GB",  # Each shard max 2 GB
    filename_pattern="model-{:05d}-of-{:05d}.safetensors"
)

# Save shards
for shard_file, shard in shards.items():
    save(shard, os.path.join(output_dir, shard_file))

# Save index (maps parameters to shards)
with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f)
```

**Benefits:**
- Parallel loading (load shards concurrently)
- Reduced memory usage during load
- Better for cloud storage (smaller files)

---

## Complete Example: Production Checkpointing

```python
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import os
import shutil

# Configure checkpointing
project_config = ProjectConfiguration(
    project_dir="./training_run",
    automatic_checkpoint_naming=True,
    total_limit=3,  # Keep only 3 checkpoints
    iteration=0,
)

accelerator = Accelerator(
    project_config=project_config,
    mixed_precision="fp16",
    gradient_accumulation_steps=4,
)

model, optimizer, dataloader, scheduler = accelerator.prepare(...)

# Training loop with checkpointing
best_loss = float('inf')
checkpoint_every_n_epochs = 5

for epoch in range(100):
    # Train
    model.train()
    for batch in dataloader:
        with accelerator.accumulate(model):
            loss = compute_loss(model, batch)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step()

    # Evaluate
    eval_loss = evaluate(model, val_dataloader)

    # Save checkpoint every N epochs
    if (epoch + 1) % checkpoint_every_n_epochs == 0:
        accelerator.save_state()
        accelerator.wait_for_everyone()

    # Save best checkpoint separately
    if eval_loss < best_loss and accelerator.is_main_process:
        best_loss = eval_loss
        best_dir = os.path.join(accelerator.project_dir, "best_checkpoint")
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        # Save state to temporary location, then copy to best
        accelerator.save_state(best_dir)

    accelerator.wait_for_everyone()

# Load best checkpoint for final evaluation
accelerator.load_state(os.path.join(accelerator.project_dir, "best_checkpoint"))
final_eval(model)
```

---

## Summary

**Checkpoint Components:**
- Model weights
- Optimizer states (2× model size for Adam/AdamW)
- LR scheduler state
- DataLoader sampler state
- GradScaler state (FP16)
- RNG states (all sources)
- Step counter
- Custom objects

**Key Functions:**
- `accelerator.save_state(output_dir)`
- `accelerator.load_state(input_dir)`
- `accelerator.register_for_checkpointing(obj)`

**Best Practices:**
1. Use `wait_for_everyone()` after save
2. Enable `safe_serialization=True` (safetensors)
3. Save to fast local storage
4. Validate checkpoints after save
5. Use `total_limit` for automatic rotation
6. Save best checkpoint separately
7. Test checkpoint resume regularly

**Backend-Specific:**
- **DDP:** Unwraps model, saves standard state_dict
- **FSDP:** Gathers sharded parameters before save
- **DeepSpeed:** Uses custom ZeRO checkpoint format
- **Megatron-LM:** Handles model + optimizer + scheduler together

**Reproducibility:**
- Accelerate saves ALL RNG states (Python, NumPy, PyTorch, CUDA, XLA)
- Loading checkpoint restores exact random state
- Enables bit-exact reproducibility for debugging

---

## Related Files
- **Checkpointing module:** `src/accelerate/checkpointing.py`
- **save_state implementation:** `src/accelerate/accelerator.py:3467`
- **load_state implementation:** `src/accelerate/accelerator.py:3633`
- **FSDP utilities:** `src/accelerate/utils/fsdp_utils.py`
- **Test coverage:** `tests/test_state_checkpointing.py`
