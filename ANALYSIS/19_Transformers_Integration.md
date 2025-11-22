# Integration 8: Transformers Integration

## Overview

HuggingFace Transformers has native Accelerate integration via `device_map` parameter.

---

## Single-Line Loading

```python
from transformers import AutoModelForCausalLM

# Accelerate integration built-in!
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    max_memory={0: "20GB", 1: "20GB", "cpu": "100GB"},
    torch_dtype=torch.float16,
    load_in_8bit=True,  # BitsAndBytes integration
)
```

**Internally calls:**
1. `init_empty_weights()`
2. `infer_auto_device_map()`
3. `load_checkpoint_and_dispatch()`

---

## Key Parameters

### device_map

```python
device_map="auto"          # Auto-distribute
device_map="balanced"      # Equal GPU distribution
device_map="sequential"    # Fill GPUs sequentially
device_map={"layer.0": 0}  # Manual mapping
```

### max_memory

```python
max_memory={0: "20GB", "cpu": "100GB"}
```

### torch_dtype

```python
torch_dtype=torch.float16    # FP16
torch_dtype=torch.bfloat16   # BF16
torch_dtype="auto"           # Match checkpoint dtype
```

### Quantization

```python
load_in_8bit=True   # 8-bit quantization
load_in_4bit=True   # 4-bit quantization
```

---

## Examples

### Multi-GPU Inference

```python
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    device_map="auto",  # Uses all GPUs
)
```

### CPU Offload

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    device_map="auto",
    max_memory={0: "10GB", "cpu": "80GB"},  # Heavy CPU usage
)
```

### 4-bit Loading

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

---

Next: **Group 3 - BitsAndBytes Quantization**
