# Integration 6: Device Map Strategies Guide

## Overview

Device maps control how model layers are distributed across GPUs, CPU, and disk. Accelerate provides automatic strategies and manual control for optimal memory usage.

**File:** `src/accelerate/utils/modeling.py` (infer_auto_device_map function)

---

## Auto Strategies

### 1. "auto" (Recommended)

```python
device_map = "auto"  # Sequential fill: GPU0 → GPU1 → CPU → Disk
```

**Algorithm:**
1. Calculate memory per layer
2. Fill GPU 0 until `max_memory[0]` reached
3. Fill GPU 1, then CPU, then disk
4. Never split layers (respects `no_split_module_classes`)

---

### 2. "balanced"

```python
device_map = "balanced"  # Equal distribution across GPUs
```

**Algorithm:** Distributes layers evenly across all GPUs first, then CPU.

---

### 3. "balanced_low_0"

```python
device_map = "balanced_low_0"  # Minimize GPU 0 usage
```

**Use case:** GPU 0 handles inputs, leave more space for activations.

---

### 4. "sequential"

```python
device_map = "sequential"  # Same as "auto"
```

---

## Manual Device Maps

```python
device_map = {
    "model.embed_tokens": 0,          # GPU 0
    "model.layers.0": 0,
    "model.layers.1": 1,              # GPU 1
    "model.layers.2": "cpu",          # CPU
    "model.layers.3": "disk",         # Disk
    "lm_head": "cpu",
}
```

**Key rules:**
- Every module must have a device
- Use integers (0, 1, 2) for GPUs
- Use strings ("cpu", "disk") for CPU/disk

---

## Memory Specification

```python
max_memory = {
    0: "20GB",           # GPU 0: 20GB
    1: "20GB",           # GPU 1: 20GB
    "cpu": "100GB",      # CPU: 100GB
    "disk": "500GB",     # Disk (optional)
}
```

**Tips:**
- Leave 15-20% headroom for activations
- 24GB GPU → specify "20GB"
- Activations scale with batch size

---

## no_split_module_classes

```python
device_map = infer_auto_device_map(
    model,
    max_memory={0: "20GB", "cpu": "100GB"},
    no_split_module_classes=["LlamaDecoderLayer"],  # Keep intact
)
```

**Common classes:**
- Llama: `["LlamaDecoderLayer"]`
- BLOOM: `["BloomBlock"]`
- GPT-J: `["GPTJBlock"]`
- GPT-2: `["GPT2Block"]`
- T5: `["T5Block"]`

---

## Optimization Strategies

### For Inference (Single Sample)

```python
# Maximize GPU usage, minimize CPU
max_memory = {0: "22GB", 1: "22GB", "cpu": "20GB"}
```

---

### For Training (Batch Processing)

```python
# Leave more space for gradients and optimizer states
max_memory = {0: "16GB", 1: "16GB", "cpu": "100GB"}
```

---

### For Multi-User Server

```python
# Balanced approach
device_map = "balanced_low_0"  # Leave GPU 0 less loaded
```

---

## File References

- `infer_auto_device_map()`: `src/accelerate/utils/modeling.py` (~line 400+)
- `check_device_map()`: Validation logic
- `get_balanced_memory()`: Memory calculation

---

Next: **18_CPU_Offloading_Patterns.md**
