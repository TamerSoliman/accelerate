# Integration 5: Large Model Loading Tutorial

## Overview

Accelerate's `big_modeling` module enables loading models that don't fit in GPU memory by:
1. **Meta device initialization** - Create model structure without allocating memory
2. **Device mapping** - Distribute layers across GPUs, CPU, and disk
3. **Checkpoint loading** - Load weights directly to target devices
4. **CPU offloading** - Keep weights on CPU, move to GPU only during forward pass

**Use Case:** Load 70B+ parameter models on consumer hardware (e.g., 24GB GPU + 128GB RAM).

**File Location:** `src/accelerate/big_modeling.py`

---

## Problem: Models Too Large for Single GPU

```python
# ❌ This FAILS for 70B models on 24GB GPU
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")
# OOM: Requires ~140GB for FP32, ~70GB for FP16
```

**Why it fails:**
- **70B parameters × 2 bytes (FP16) = 140GB**
- Single 24GB GPU cannot hold model
- Standard loading allocates ALL weights first

---

## Solution 1: Meta Device + Device Map

### Step 1: Init on Meta Device (Zero Memory)

```python
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

# Load config (tiny, ~KB)
config = AutoConfig.from_pretrained("meta-llama/Llama-2-70b-hf")

# Initialize model structure WITHOUT allocating memory
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# Model structure exists, but parameters are on "meta" device
print(model.lm_head.weight.device)  # device(type='meta')
print(model.lm_head.weight.shape)   # torch.Size([32000, 8192])
```

**Key Insight:** `init_empty_weights()` creates model architecture with NO memory allocation.

---

### Step 2: Create Device Map

```python
from accelerate import infer_auto_device_map

# Automatically distribute layers across available devices
device_map = infer_auto_device_map(
    model,
    max_memory={
        0: "20GB",  # GPU 0: reserve 20GB (leave 4GB for activations)
        1: "20GB",  # GPU 1
        "cpu": "100GB",  # CPU: 100GB for remaining layers
    },
    no_split_module_classes=["LlamaDecoderLayer"],  # Keep layers intact
)

print(device_map)
# Output:
# {
#     'model.embed_tokens': 0,
#     'model.layers.0': 0,
#     'model.layers.1': 0,
#     ...
#     'model.layers.40': 0,   # GPU 0 full
#     'model.layers.41': 1,   # Start GPU 1
#     ...
#     'model.layers.70': 1,   # GPU 1 full
#     'model.layers.71': 'cpu',  # Remaining on CPU
#     ...
#     'lm_head': 'cpu',
# }
```

**What it does:**
- Calculates memory required per layer
- Fills GPU 0, then GPU 1, then CPU
- Never splits layers (keeps `LlamaDecoderLayer` intact)

---

### Step 3: Load Checkpoint with Dispatch

```python
from accelerate import load_checkpoint_and_dispatch

# Load weights directly to devices specified in device_map
model = load_checkpoint_and_dispatch(
    model,
    checkpoint="meta-llama/Llama-2-70b-hf",  # HuggingFace Hub or local path
    device_map=device_map,
)

# Now model is ready for inference!
print(model.lm_head.weight.device)  # device(type='cpu')
print(model.model.layers[0].self_attn.q_proj.weight.device)  # device(type='cuda', index=0)
```

**What happens:**
1. Loads checkpoint shards (e.g., `pytorch_model-00001-of-00015.bin`)
2. For each weight, places it on device from `device_map`
3. Never loads entire model into memory at once

---

## Complete Example: Load 70B Model

```python
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

# Step 1: Create empty model
config = AutoConfig.from_pretrained("meta-llama/Llama-2-70b-hf")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# Step 2: Create device map
device_map = infer_auto_device_map(
    model,
    max_memory={
        0: "20GB",
        1: "20GB",
        "cpu": "100GB",
    },
    no_split_module_classes=["LlamaDecoderLayer"],
)

# Step 3: Load and dispatch
model = load_checkpoint_and_dispatch(
    model,
    checkpoint="meta-llama/Llama-2-70b-hf",
    device_map=device_map,
)

# Step 4: Run inference
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda:0")

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)

print(tokenizer.decode(outputs[0]))
```

**Memory Usage:**
- GPU 0: ~20GB (layers 0-40)
- GPU 1: ~20GB (layers 41-70)
- CPU: ~30GB (final layers + embeddings)
- **Total: 70GB distributed across devices**

---

## Transformers Integration

Transformers has built-in support for Accelerate:

```python
from transformers import AutoModelForCausalLM

# Single line! Transformers handles init_empty_weights + dispatch internally
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",  # Automatically distribute
    max_memory={0: "20GB", 1: "20GB", "cpu": "100GB"},
)

# Model ready to use!
```

**How it works:**
1. Transformers detects `device_map="auto"`
2. Calls `init_empty_weights()` internally
3. Calls `infer_auto_device_map()`
4. Calls `load_checkpoint_and_dispatch()`

---

## Device Map Strategies

### Strategy 1: Auto (Recommended)

```python
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-176b",
    device_map="auto",  # Automatically balance across GPUs
)
```

**Behavior:**
- Fills GPU 0, then GPU 1, then CPU, then disk
- Balances memory usage
- No manual configuration needed

---

### Strategy 2: Balanced

```python
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-176b",
    device_map="balanced",  # Equal distribution across GPUs
)
```

**Behavior:**
- Distributes layers evenly across all GPUs
- May not be optimal (some layers larger than others)

---

### Strategy 3: Balanced Low 0

```python
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-176b",
    device_map="balanced_low_0",  # Minimize GPU 0 usage
)
```

**Behavior:**
- Leaves more space on GPU 0 for activations
- Useful when GPU 0 also runs input processing

---

### Strategy 4: Sequential

```python
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-176b",
    device_map="sequential",  # Fill GPU 0 completely, then GPU 1, etc.
)
```

**Behavior:**
- Fills devices sequentially
- Same as "auto" but explicit

---

### Strategy 5: Custom

```python
# Manual control
device_map = {
    "model.embed_tokens": "cpu",
    "model.layers.0": 0,
    "model.layers.1": 0,
    # ... specify every layer
    "lm_head": "cpu",
}

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map=device_map,
)
```

---

## CPU Offloading for Inference

For **inference-only** (no training), use aggressive CPU offloading:

```python
from accelerate import cpu_offload

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")

# Offload ALL layers to CPU
cpu_offload(
    model,
    execution_device=torch.device("cuda:0"),  # Run forward pass on GPU 0
)

# Inference works, but SLOW (weights copied GPU ↔ CPU each forward pass)
outputs = model.generate(**inputs, max_length=50)
```

**How it works:**
1. All weights stored on CPU
2. During forward pass, copy layer to GPU
3. Execute layer
4. Copy layer back to CPU
5. Repeat for next layer

**Trade-off:** Saves memory, but 10-100x slower due to CPU ↔ GPU transfers.

---

## Disk Offloading (Ultra-Low Memory)

For models that don't even fit in CPU RAM:

```python
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-176b",  # 352GB model
    device_map="auto",
    max_memory={0: "20GB", "cpu": "50GB"},  # Limited RAM
    offload_folder="./offload",  # Offload to disk
)
```

**Behavior:**
- Weights that don't fit in GPU or CPU are saved to disk
- Loaded on-demand during forward pass
- **VERY SLOW** (disk I/O bottleneck)

**Use case:** Running 176B model on consumer hardware for testing, not production.

---

## Memory Optimization Tips

### 1. Use Lower Precision

```python
# FP16 saves 50% memory
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    torch_dtype=torch.float16,  # 70GB → 35GB
)

# BF16 (better numerical stability)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

---

### 2. 8-bit Quantization (BitsAndBytes)

```python
# Load in 8-bit (87.5% memory reduction)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    load_in_8bit=True,  # 70GB → ~9GB
)
```

**Savings:**
- FP32: 4 bytes/param
- FP16: 2 bytes/param
- INT8: 1 byte/param

---

### 3. 4-bit Quantization (QLoRA)

```python
# Load in 4-bit (93.75% memory reduction)
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    quantization_config=bnb_config,  # 70GB → ~5GB!
)
```

---

## Common Patterns

### Pattern 1: Multi-GPU Inference

```python
# Automatically use all available GPUs
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",  # Detects all GPUs
    torch_dtype=torch.float16,
)

# Inputs go to first device
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
outputs = model.generate(**inputs, max_length=100)
```

---

### Pattern 2: Check Device Placement

```python
# See where each layer is placed
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")

# Output:
# model.embed_tokens.weight: cuda:0
# model.layers.0.self_attn.q_proj.weight: cuda:0
# ...
# model.layers.40.self_attn.q_proj.weight: cuda:1
# ...
# lm_head.weight: cpu
```

---

### Pattern 3: Save Device Map for Reuse

```python
import json

# Save device map
with open("device_map.json", "w") as f:
    json.dump(device_map, f)

# Load and reuse
with open("device_map.json", "r") as f:
    device_map = json.load(f)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map=device_map,
)
```

---

## Troubleshooting

### Issue 1: OOM Even with device_map="auto"

**Problem:** Still running out of memory.

**Solution:** Reduce `max_memory` to leave headroom for activations:

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    max_memory={
        0: "18GB",  # Leave 6GB for activations (was 24GB total)
        "cpu": "80GB",
    },
)
```

---

### Issue 2: Slow Inference

**Problem:** Model runs but very slow.

**Cause:** Layers on CPU or disk.

**Solution:** Use more GPUs or quantization:

```python
# Option 1: Add more GPUs
max_memory={0: "20GB", 1: "20GB", 2: "20GB"}

# Option 2: Use quantization
load_in_8bit=True  # Fit more on GPU
```

---

### Issue 3: Layer Splitting Errors

**Problem:** `RuntimeError: Expected all tensors to be on the same device`

**Cause:** Layer split across devices.

**Solution:** Specify `no_split_module_classes`:

```python
device_map = infer_auto_device_map(
    model,
    max_memory={0: "20GB", "cpu": "100GB"},
    no_split_module_classes=["LlamaDecoderLayer", "BloomBlock", "GPTJBlock"],
)
```

---

## Summary

### Key Functions

| Function | Purpose | Usage |
|----------|---------|-------|
| `init_empty_weights()` | Create model without memory | `with init_empty_weights(): model = Model()` |
| `infer_auto_device_map()` | Auto-generate device mapping | `device_map = infer_auto_device_map(model, ...)` |
| `load_checkpoint_and_dispatch()` | Load weights to devices | `model = load_checkpoint_and_dispatch(model, ...)` |
| `cpu_offload()` | Offload all weights to CPU | `cpu_offload(model, execution_device=...)` |

### Memory Strategies

| Strategy | Memory Reduction | Quality | Speed |
|----------|------------------|---------|-------|
| FP16 | 50% | Excellent | Fast |
| 8-bit | 75% | Very Good | Medium |
| 4-bit | 87% | Good | Medium |
| CPU offload | 90%+ | Excellent | Very Slow |
| Disk offload | 95%+ | Excellent | Extremely Slow |

### File References

- **init_empty_weights**: `src/accelerate/big_modeling.py:60-93`
- **load_checkpoint_and_dispatch**: `src/accelerate/big_modeling.py` (function at ~line 400+)
- **dispatch_model**: `src/accelerate/big_modeling.py:309-408`
- **cpu_offload**: `src/accelerate/big_modeling.py:173-208`

### Next Steps

- **17_Device_Map_Strategies_Guide.md** - Deep dive on device mapping algorithms
- **18_CPU_Offloading_Patterns.md** - CPU offload for inference optimization
- **19_Transformers_Integration.md** - Integration with HuggingFace Transformers
