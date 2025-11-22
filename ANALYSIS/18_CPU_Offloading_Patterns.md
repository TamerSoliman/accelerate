# Integration 7: CPU Offloading Patterns

## Overview

CPU offloading keeps model weights on CPU, moving layers to GPU only during forward pass. Trades memory for speed.

**File:** `src/accelerate/big_modeling.py:173-208` (cpu_offload function)

---

## Basic Pattern

```python
from accelerate import cpu_offload
import torch

# Model on CPU
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Enable offloading
cpu_offload(model, execution_device=torch.device("cuda:0"))

# Inference works (slow)
outputs = model(**inputs)
```

**How it works:**
1. All weights stored on CPU
2. Before layer forward: copy layer to GPU
3. Execute layer on GPU
4. After layer forward: copy layer back to CPU

---

## Performance Trade-offs

| Strategy | GPU Memory | Speed | Use Case |
|----------|-----------|-------|----------|
| No offload | High | Fast | Standard |
| CPU offload | Low | 10-50x slower | Memory-constrained |
| Disk offload | Minimal | 100x+ slower | Testing only |

---

## Selective Offloading

```python
# Offload only specific layers
from accelerate.hooks import add_hook_to_module, CpuOffload

for i in range(20, 40):  # Offload layers 20-39
    layer = model.model.layers[i]
    add_hook_to_module(layer, CpuOffload(execution_device="cuda:0"))
```

---

## File References

- `cpu_offload()`: `src/accelerate/big_modeling.py:173-208`
- `CpuOffload` hook: `src/accelerate/hooks.py`

---

Next: **19_Transformers_Integration.md**
