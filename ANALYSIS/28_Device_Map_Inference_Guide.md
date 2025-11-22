# Integration 17: Device Map Inference Guide

## infer_auto_device_map() Internals

**File:** `src/accelerate/utils/modeling.py`

---

## Algorithm

```python
def infer_auto_device_map(model, max_memory, no_split_module_classes):
    # 1. Calculate memory per parameter
    param_sizes = {}
    for name, param in model.named_parameters():
        param_sizes[name] = param.numel() * param.element_size()

    # 2. Group by modules
    module_sizes = aggregate_by_module(param_sizes)

    # 3. Allocate to devices
    device_map = {}
    current_device = 0
    current_memory = 0

    for module_name, size in module_sizes.items():
        if current_memory + size > max_memory[current_device]:
            current_device += 1
            current_memory = 0

        device_map[module_name] = current_device
        current_memory += size

    return device_map
```

---

## Memory Calculation

```python
# Per parameter: numel × dtype_size
# Example: (1024, 1024) FP16 tensor
memory = 1024 * 1024 * 2 bytes = 2 MB
```

---

## Respecting no_split_module_classes

```python
# Ensures entire module stays on one device
no_split_module_classes = ["LlamaDecoderLayer"]

# If layer doesn't fit on current device → move to next device
```

---

Next: **29_Sharded_Checkpoint_Tutorial.md**
