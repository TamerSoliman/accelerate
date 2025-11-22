# Configuration Reference: Accelerate Parameters & Environment Variables

## Overview
This guide documents the critical configuration parameters that control Accelerate's distributed setup. Configuration can be specified via:
1. **YAML/JSON config file** (created with `accelerate config`)
2. **Environment variables** (set by `accelerate launch` or manually)
3. **Programmatic parameters** (passed to `Accelerator()`)

---

## Table 1: Core Configuration Parameters

| **Parameter** | **Type** | **Default** | **Purpose** | **Impact on Distributed Setup** | **File Location** |
|--------------|----------|------------|------------|--------------------------------|-------------------|
| **compute_environment** | `ComputeEnvironment` | `LOCAL_MACHINE` | Specifies where training runs (local machine, AWS SageMaker, etc.) | Determines if using cloud-specific optimizations | `src/accelerate/commands/config/config_args.py:77` |
| **distributed_type** | `DistributedType` | Auto-detected | Backend for distributed training: `NO`, `MULTI_GPU`, `MULTI_CPU`, `DEEPSPEED`, `FSDP`, `MEGATRON_LM`, `XLA` | **CRITICAL:** Determines which wrapping strategy is used (DDP, FSDP, DeepSpeed, etc.) | `src/accelerate/utils/dataclasses.py:DistributedType` |
| **num_processes** | `int` | 1 | Total number of processes to launch across all nodes | Sets `WORLD_SIZE`, controls data sharding, gradient averaging | `src/accelerate/state.py:PartialState` |
| **num_machines** | `int` | 1 | Number of nodes (machines) in the cluster | Used to calculate ranks and set up multi-node communication | `src/accelerate/state.py` |
| **machine_rank** | `int` | 0 | Rank of the current machine (0 for main node) | Determines which machine is the coordinator (rank 0) | `src/accelerate/state.py` |
| **main_process_ip** | `str` | `localhost` | IP address of the main process (rank 0) | Set as `MASTER_ADDR` for `torch.distributed.init_process_group()` | `src/accelerate/utils/launch.py` |
| **main_process_port** | `int` | 29500 | Port for inter-process communication | Set as `MASTER_PORT` for `torch.distributed.init_process_group()` | `src/accelerate/utils/launch.py` |
| **mixed_precision** | `str` | `"no"` | Mixed precision mode: `"no"`, `"fp16"`, `"bf16"`, `"fp8"` | Enables GradScaler (FP16), autocast contexts, impacts memory and speed | `src/accelerate/accelerator.py:282` |
| **gradient_accumulation_steps** | `int` | 1 | Number of steps to accumulate gradients before optimizer update | Reduces memory usage (smaller per-GPU batch), increases effective batch size | `src/accelerate/accelerator.py:283` |
| **use_cpu** | `bool` | `False` | Force training on CPU (ignore GPUs) | Overrides device detection, useful for debugging or CPU-only environments | `src/accelerate/accelerator.py:284` |

---

## Table 2: Backend-Specific Plugins

### DeepSpeed Plugin (`deepspeed_config`)

| **Parameter** | **Type** | **Default** | **Purpose** | **File Location** |
|--------------|----------|------------|------------|-------------------|
| **zero_stage** | `int` | 0 | ZeRO optimization stage (0, 1, 2, 3) | `src/accelerate/utils/dataclasses.py:DeepSpeedPlugin` |
| | | | - **Stage 0:** No sharding (standard DDP) | |
| | | | - **Stage 1:** Shard optimizer states | |
| | | | - **Stage 2:** Shard optimizer states + gradients | |
| | | | - **Stage 3:** Shard optimizer states + gradients + parameters | |
| **offload_optimizer_device** | `str` | `"none"` | Offload optimizer states to CPU: `"none"`, `"cpu"`, `"nvme"` | `src/accelerate/utils/dataclasses.py:DeepSpeedPlugin` |
| **offload_param_device** | `str` | `"none"` | Offload parameters to CPU/NVMe (ZeRO-3 only) | `src/accelerate/utils/dataclasses.py:DeepSpeedPlugin` |
| **gradient_clipping** | `float` | `None` | Max gradient norm for clipping (handled by DeepSpeed) | `src/accelerate/utils/dataclasses.py:DeepSpeedPlugin` |
| **gradient_accumulation_steps** | `int` | Inherited | Overrides global gradient accumulation for DeepSpeed | `src/accelerate/utils/dataclasses.py:DeepSpeedPlugin` |

**Impact:** DeepSpeed plugin completely changes the model/optimizer wrapping process. Instead of DDP, the model is wrapped with `DeepSpeedEngine`, which manages all training state.

**Code Path:** `src/accelerate/accelerator.py:2064` (_prepare_deepspeed)

---

### FSDP Plugin (`fsdp_config`)

| **Parameter** | **Type** | **Default** | **Purpose** | **File Location** |
|--------------|----------|------------|------------|-------------------|
| **fsdp_version** | `int` | 1 | FSDP version: `1` (legacy) or `2` (PyTorch 2.6+) | `src/accelerate/utils/dataclasses.py:FullyShardedDataParallelPlugin` |
| **sharding_strategy** | `ShardingStrategy` | `FULL_SHARD` | Sharding strategy: | `src/accelerate/utils/dataclasses.py:FullyShardedDataParallelPlugin` |
| | | | - **FULL_SHARD:** Shard params, gradients, optimizer states | |
| | | | - **SHARD_GRAD_OP:** Shard gradients and optimizer states only | |
| | | | - **NO_SHARD:** No sharding (equivalent to DDP) | |
| | | | - **HYBRID_SHARD:** Shard within nodes, replicate across nodes | |
| **cpu_offload** | `CPUOffload` | `None` | Offload parameters to CPU during backward pass | `src/accelerate/utils/dataclasses.py:FullyShardedDataParallelPlugin` |
| **auto_wrap_policy** | `callable` | `None` | Policy for automatically wrapping submodules (e.g., `transformer_auto_wrap_policy`) | `src/accelerate/utils/dataclasses.py:FullyShardedDataParallelPlugin` |
| **backward_prefetch** | `BackwardPrefetch` | `BACKWARD_PRE` | Prefetch strategy for backward pass: | `src/accelerate/utils/dataclasses.py:FullyShardedDataParallelPlugin` |
| | | | - **BACKWARD_PRE:** Prefetch next layer's parameters during backward | |
| | | | - **BACKWARD_POST:** Prefetch after current layer's backward | |
| **cpu_ram_efficient_loading** | `bool` | `False` | Load model weights directly to GPU, bypass CPU (requires PyTorch 2.0+) | `src/accelerate/utils/dataclasses.py:FullyShardedDataParallelPlugin` |

**Impact:** FSDP plugin enables training models larger than GPU memory by sharding parameters across GPUs. Memory usage is reduced from N × model_size (DDP) to approximately model_size (FSDP).

**Code Path:** `src/accelerate/accelerator.py:1843` (FSDP1) or `src/accelerate/accelerator.py:1633` (FSDP2)

---

### Megatron-LM Plugin (`megatron_lm_config`)

| **Parameter** | **Type** | **Default** | **Purpose** | **File Location** |
|--------------|----------|------------|------------|-------------------|
| **tp_degree** | `int` | 1 | Tensor parallelism degree (split model across GPUs) | `src/accelerate/utils/dataclasses.py:MegatronLMPlugin` |
| **pp_degree** | `int` | 1 | Pipeline parallelism degree (split layers across GPUs) | `src/accelerate/utils/dataclasses.py:MegatronLMPlugin` |
| **num_micro_batches** | `int` | 1 | Number of micro-batches for pipeline parallelism | `src/accelerate/utils/dataclasses.py:MegatronLMPlugin` |

**Impact:** Megatron-LM enables training extremely large models (100B+ parameters) using tensor and pipeline parallelism. Requires Megatron-LM library.

**Code Path:** `src/accelerate/accelerator.py` (_prepare_megatron_lm)

---

## Table 3: Mixed Precision & Optimization

### GradScalerKwargs (FP16 Scaling)

| **Parameter** | **Type** | **Default** | **Purpose** | **File Location** |
|--------------|----------|------------|------------|-------------------|
| **init_scale** | `float` | 65536.0 | Initial loss scale factor | `src/accelerate/utils/dataclasses.py:GradScalerKwargs` |
| **growth_factor** | `float` | 2.0 | Factor by which to increase scale when no overflow | `src/accelerate/utils/dataclasses.py:GradScalerKwargs` |
| **backoff_factor** | `float` | 0.5 | Factor by which to decrease scale when overflow detected | `src/accelerate/utils/dataclasses.py:GradScalerKwargs` |
| **growth_interval** | `int` | 2000 | Number of steps before attempting to increase scale | `src/accelerate/utils/dataclasses.py:GradScalerKwargs` |
| **enabled** | `bool` | `True` | Enable/disable scaler (useful for debugging) | `src/accelerate/utils/dataclasses.py:GradScalerKwargs` |

**Usage:**
```python
from accelerate import Accelerator
from accelerate.utils import GradScalerKwargs

scaler_kwargs = GradScalerKwargs(init_scale=32768.0, growth_interval=1000)
accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[scaler_kwargs])
```

**Impact:** Controls the dynamic loss scaling for FP16 training. Larger `init_scale` provides more precision but higher risk of overflow.

---

### AutocastKwargs (Mixed Precision Context)

| **Parameter** | **Type** | **Default** | **Purpose** | **File Location** |
|--------------|----------|------------|------------|-------------------|
| **enabled** | `bool` | `True` | Enable/disable autocast | `src/accelerate/utils/dataclasses.py:AutocastKwargs` |
| **cache_enabled** | `bool` | `None` | Enable autocast weight caching (PyTorch 2.0+) | `src/accelerate/utils/dataclasses.py:AutocastKwargs` |

**Usage:**
```python
from accelerate.utils import AutocastKwargs

autocast_kwargs = AutocastKwargs(cache_enabled=True)
accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[autocast_kwargs])
```

---

### DistributedDataParallelKwargs (DDP Configuration)

| **Parameter** | **Type** | **Default** | **Purpose** | **File Location** |
|--------------|----------|------------|------------|-------------------|
| **bucket_cap_mb** | `int` | 25 | Max size of DDP communication bucket (MB) | `src/accelerate/utils/dataclasses.py:DistributedDataParallelKwargs` |
| **find_unused_parameters** | `bool` | `False` | Find and exclude unused parameters from gradient sync | `src/accelerate/utils/dataclasses.py:DistributedDataParallelKwargs` |
| **gradient_as_bucket_view** | `bool` | `False` | Use bucket view for gradients (saves memory) | `src/accelerate/utils/dataclasses.py:DistributedDataParallelKwargs` |
| **static_graph** | `bool` | `False` | Optimize for static computation graph (PyTorch 1.11+) | `src/accelerate/utils/dataclasses.py:DistributedDataParallelKwargs` |

**Usage:**
```python
from accelerate.utils import DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(
    find_unused_parameters=True,
    gradient_as_bucket_view=True
)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
```

**Impact:**
- `bucket_cap_mb`: Larger buckets reduce communication overhead but increase memory
- `find_unused_parameters=True`: Required for models with conditional branches (e.g., GANs)
- `gradient_as_bucket_view=True`: Reduces memory by ~10% (recommended for most cases)

---

## Table 4: Critical Environment Variables

These are set automatically by `accelerate launch` or `torchrun`:

| **Environment Variable** | **Purpose** | **Example Value** | **Set By** | **Read By** |
|-------------------------|-----------|------------------|-----------|-----------|
| `WORLD_SIZE` | Total number of processes across all nodes | `4` (for 4 GPUs) | `accelerate launch` | `src/accelerate/state.py:PartialState.__init__` |
| `RANK` | Global rank of current process (0 to WORLD_SIZE-1) | `0`, `1`, `2`, `3` | `accelerate launch` | `src/accelerate/state.py:PartialState.__init__` |
| `LOCAL_RANK` | Local rank on current node (0 to num_processes_per_node-1) | `0`, `1`, `2`, `3` | `accelerate launch` | `src/accelerate/state.py:PartialState.__init__` |
| `MASTER_ADDR` | IP address of rank 0 process | `192.168.1.100` | `accelerate launch` | `torch.distributed.init_process_group()` |
| `MASTER_PORT` | Port for inter-process communication | `29500` | `accelerate launch` | `torch.distributed.init_process_group()` |
| `ACCELERATE_MIXED_PRECISION` | Mixed precision mode: `no`, `fp16`, `bf16`, `fp8` | `fp16` | `accelerate launch` | `src/accelerate/accelerator.py:282` |
| `ACCELERATE_GRADIENT_ACCUMULATION_STEPS` | Number of gradient accumulation steps | `4` | `accelerate launch` | `src/accelerate/accelerator.py:545` |
| `ACCELERATE_USE_DEEPSPEED` | Enable DeepSpeed backend | `true` | `accelerate launch` | `src/accelerate/accelerator.py:346` |
| `ACCELERATE_USE_FSDP` | Enable FSDP backend | `true` | `accelerate launch` | `src/accelerate/accelerator.py:380` |
| `ACCELERATE_USE_MEGATRON_LM` | Enable Megatron-LM backend | `true` | `accelerate launch` | `src/accelerate/accelerator.py:402` |

**How to Set Manually:**
```bash
# For 4 GPUs on a single node:
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Launch 4 processes:
RANK=0 LOCAL_RANK=0 python train.py &
RANK=1 LOCAL_RANK=1 python train.py &
RANK=2 LOCAL_RANK=2 python train.py &
RANK=3 LOCAL_RANK=3 python train.py &
```

**How accelerate launch Sets These:**
```bash
# accelerate launch --num_processes=4 --mixed_precision=fp16 train.py

# Internally creates:
WORLD_SIZE=4
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
ACCELERATE_MIXED_PRECISION=fp16

# Then spawns 4 processes with RANK=0,1,2,3 and LOCAL_RANK=0,1,2,3
```

---

## Table 5: DataLoader Configuration

| **Parameter** | **Type** | **Default** | **Purpose** | **Impact** | **File Location** |
|--------------|----------|------------|------------|-----------|-------------------|
| **split_batches** | `bool` | `False` | Split batches across processes (vs. replicate batch_size per process) | `True`: effective_batch = batch_size, `False`: effective_batch = batch_size × num_processes | `src/accelerate/utils/dataclasses.py:DataLoaderConfiguration` |
| **dispatch_batches** | `bool` | `False` | Main process loads data, dispatches to other processes | Useful for slow dataloaders or limited I/O bandwidth | `src/accelerate/utils/dataclasses.py:DataLoaderConfiguration` |
| **even_batches** | `bool` | `True` | Ensure all processes have same number of batches (pad if necessary) | Prevents deadlocks when different processes have different dataset sizes | `src/accelerate/utils/dataclasses.py:DataLoaderConfiguration` |
| **use_seedable_sampler** | `bool` | `False` | Use seedable random sampler for reproducibility | Ensures same data order across runs | `src/accelerate/utils/dataclasses.py:DataLoaderConfiguration` |
| **non_blocking** | `bool` | `False` | Use non-blocking transfers when moving data to device | Can improve throughput by overlapping data transfer with computation | `src/accelerate/utils/dataclasses.py:DataLoaderConfiguration` |

**Usage:**
```python
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration

dataloader_config = DataLoaderConfiguration(
    split_batches=False,  # Replicate batch_size per GPU
    even_batches=True,    # Pad to ensure same number of batches
    use_seedable_sampler=True  # For reproducibility
)

accelerator = Accelerator(dataloader_config=dataloader_config)
```

---

## Table 6: Project Configuration

| **Parameter** | **Type** | **Default** | **Purpose** | **File Location** |
|--------------|----------|------------|------------|-------------------|
| **project_dir** | `str` | `None` | Root directory for outputs (checkpoints, logs, tensorboard) | `src/accelerate/utils/dataclasses.py:ProjectConfiguration` |
| **logging_dir** | `str` | `{project_dir}/logs` | Directory for logs and tensorboard events | `src/accelerate/utils/dataclasses.py:ProjectConfiguration` |
| **automatic_checkpoint_naming** | `bool` | `False` | Automatically name checkpoints (e.g., `checkpoint-1000`) | `src/accelerate/utils/dataclasses.py:ProjectConfiguration` |
| **total_limit** | `int` | `None` | Max number of checkpoints to keep (deletes oldest) | `src/accelerate/utils/dataclasses.py:ProjectConfiguration` |

---

## Configuration File Example (YAML)

**Location:** `~/.cache/huggingface/accelerate/default_config.yaml`

```yaml
# Created with: accelerate config
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 4
num_machines: 1
machine_rank: 0
main_process_ip: null
main_process_port: null
mixed_precision: fp16
gradient_accumulation_steps: 4
use_cpu: false
debug: false

# DDP-specific (if distributed_type: MULTI_GPU)
downcast_bf16: 'no'
gpu_ids: all
rdzv_backend: static

# FSDP-specific (if distributed_type: FSDP)
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: BertLayer

# DeepSpeed-specific (if distributed_type: DEEPSPEED)
deepspeed_config:
  deepspeed_config_file: ds_config.json
  zero_stage: 2
  offload_optimizer_device: cpu
  offload_param_device: none
  gradient_accumulation_steps: 4
  gradient_clipping: 1.0

# Logging
log_with: tensorboard
project_dir: ./outputs
logging_dir: ./outputs/logs
```

**To load this config:**
```bash
# Automatically uses ~/.cache/huggingface/accelerate/default_config.yaml
accelerate launch train.py

# Or specify custom config:
accelerate launch --config_file custom_config.yaml train.py
```

**To load programmatically:**
```python
from accelerate import Accelerator

# Uses default config from environment
accelerator = Accelerator()

# Or override specific parameters:
accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=4,
    # Other params override config file
)
```

---

## Configuration Precedence (Highest to Lowest)

1. **Programmatic arguments** passed to `Accelerator(...)`
2. **Environment variables** (e.g., `ACCELERATE_MIXED_PRECISION`)
3. **Config file** (`default_config.yaml` or `--config_file`)
4. **Default values** in code

**Example:**
```python
# Config file: mixed_precision: bf16
# Environment: ACCELERATE_MIXED_PRECISION=fp16
# Code: Accelerator(mixed_precision="fp8")

# Result: Uses "fp8" (code wins)
```

---

## Key Files for Configuration

| **Config Type** | **File Path** |
|----------------|---------------|
| **Config Loading** | `src/accelerate/commands/config/config_args.py` |
| **Dataclass Definitions** | `src/accelerate/utils/dataclasses.py` |
| **Environment Parsing** | `src/accelerate/utils/environment.py` |
| **Launch Utilities** | `src/accelerate/utils/launch.py` |
| **AcceleratorState** | `src/accelerate/state.py` |

---

## Quick Reference: Common Configurations

### Single GPU Training
```yaml
distributed_type: NO
mixed_precision: fp16
gradient_accumulation_steps: 1
```

### Multi-GPU DDP (4 GPUs)
```yaml
distributed_type: MULTI_GPU
num_processes: 4
mixed_precision: fp16
gradient_accumulation_steps: 4
```

### FSDP for Large Models
```yaml
distributed_type: FSDP
num_processes: 8
mixed_precision: bf16
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_cpu_ram_efficient_loading: true
```

### DeepSpeed ZeRO-3 with CPU Offload
```yaml
distributed_type: DEEPSPEED
num_processes: 8
deepspeed_config:
  zero_stage: 3
  offload_optimizer_device: cpu
  offload_param_device: cpu
```

---

## Summary

**10 Most Critical Parameters:**

1. **distributed_type** - Determines backend (DDP/FSDP/DeepSpeed)
2. **num_processes** - Total number of GPUs/processes
3. **mixed_precision** - FP16/BF16/FP8 for memory/speed
4. **gradient_accumulation_steps** - Effective batch size control
5. **fsdp_sharding_strategy** - FSDP memory savings (if using FSDP)
6. **deepspeed_zero_stage** - ZeRO optimization level (if using DeepSpeed)
7. **main_process_ip** - Multi-node communication
8. **main_process_port** - Multi-node communication
9. **find_unused_parameters** - DDP for models with conditional paths
10. **cpu_ram_efficient_loading** - FSDP direct-to-GPU loading

These parameters control the entire distributed setup and have the most significant impact on memory usage, training speed, and model capacity.
