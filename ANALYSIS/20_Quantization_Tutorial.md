# Integration 9: Quantization Tutorial

## Overview

BitsAndBytes integration enables 8-bit and 4-bit quantization, reducing model memory by 75-94%.

**File:** `src/accelerate/utils/bnb.py`

---

## 8-bit Quantization

### Basic Usage

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,          # Enable 8-bit
    device_map="auto",
)

# Memory: 7B params × 2 bytes (FP16) = 14GB → ~2GB (8-bit)
```

**Reduction:** 87.5% (FP16 → INT8)

---

## 4-bit Quantization (QLoRA)

### Basic Usage

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16
    bnb_4bit_quant_type="nf4",             # NormalFloat4
    bnb_4bit_use_double_quant=True,        # Nested quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Memory: 14GB (FP16) → ~1GB (4-bit)
```

**Reduction:** 93.75%

---

## Quantization Types

### NF4 (NormalFloat4) - Recommended

```python
bnb_4bit_quant_type="nf4"
```

**Best for:** Pre-trained models (weights follow normal distribution)

---

### FP4 (Float4)

```python
bnb_4bit_quant_type="fp4"
```

**Best for:** General use, slightly lower quality than NF4

---

## Compute Dtype

```python
bnb_4bit_compute_dtype=torch.float16   # FP16 compute
bnb_4bit_compute_dtype=torch.bfloat16  # BF16 compute (better)
```

**Recommendation:** Use BF16 for better numerical stability

---

## Double Quantization

```python
bnb_4bit_use_double_quant=True  # Quantize quantization constants
```

**Savings:** Additional ~0.4 bits/param reduction

---

## Training with QLoRA

```python
from peft import LoraConfig, get_peft_model

# Load base model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map="auto",
)

# Add LoRA adapters (trainable)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Train only LoRA parameters (~1% of full model)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.06%
```

---

## Memory Comparison

| Config | Memory (7B) | Memory (70B) | Quality |
|--------|-------------|--------------|---------|
| FP32   | 28GB        | 280GB        | 100%    |
| FP16   | 14GB        | 140GB        | ~100%   |
| 8-bit  | 7GB         | 70GB         | ~99%    |
| 4-bit  | 3.5GB       | 35GB         | ~95%    |

---

## File References

- **load_and_quantize_model**: `src/accelerate/utils/bnb.py:44-141`
- **BnbQuantizationConfig**: `src/accelerate/utils/dataclasses.py`

---

Next: **21_Quantization_Config_Comparison.md**
