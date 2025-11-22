# Integration 20: Memory Efficient Loading

## Complete Workflow

```python
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
)
from transformers import AutoConfig, AutoModelForCausalLM

# 1. Create empty model (0 memory)
config = AutoConfig.from_pretrained("meta-llama/Llama-2-70b-hf")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# 2. Calculate device map
device_map = infer_auto_device_map(
    model,
    max_memory={0: "20GB", 1: "20GB", "cpu": "100GB"},
    no_split_module_classes=["LlamaDecoderLayer"],
)

# 3. Load directly to target devices
model = load_checkpoint_and_dispatch(
    model,
    checkpoint="meta-llama/Llama-2-70b-hf",
    device_map=device_map,
)

# Total peak memory: ~22GB (never loads full 140GB model!)
```

---

## Key Optimizations

### 1. Meta Device Initialization

```python
with init_empty_weights():
    model = Model()  # No allocation
```

**Saves:** Entire model memory during init

---

### 2. Direct-to-Device Loading

```python
load_checkpoint_and_dispatch(model, checkpoint, device_map)
```

**Benefit:** Each shard loaded directly to target device, never loads full model in RAM

---

### 3. Offload State Dict

```python
load_checkpoint_and_dispatch(
    model,
    checkpoint,
    device_map,
    offload_state_dict=True,  # Temp save to disk
)
```

**Use case:** Model doesn't fit in RAM even temporarily

---

## Memory Timeline

```
Traditional Loading:
1. Load full model to CPU: 140GB RAM
2. Move to GPU: OOM!

Memory-Efficient Loading:
1. init_empty_weights: 0 GB
2. Load shard 1 to GPU 0: +2GB
3. Load shard 2 to GPU 0: +2GB
...
10. Load shard 40 to GPU 0: 20GB total
11. Load shard 41 to GPU 1: +2GB
...
Total peak: ~22GB (during loading)
```

---

## File References

- **init_empty_weights**: `src/accelerate/big_modeling.py:60-93`
- **init_on_device**: `src/accelerate/big_modeling.py:96-171`
- **load_checkpoint_and_dispatch**: `src/accelerate/big_modeling.py` (~line 400+)
- **load_checkpoint_in_model**: `src/accelerate/utils/modeling.py`

---

## Summary

All 20 integration documents created:
- **Group 1 (12-15):** Experiment Tracking
- **Group 2 (16-19):** Large Model Handling
- **Group 3 (20-23):** BitsAndBytes Quantization
- **Group 4 (24-27):** Hook System
- **Group 5 (28-31):** Model Utilities
