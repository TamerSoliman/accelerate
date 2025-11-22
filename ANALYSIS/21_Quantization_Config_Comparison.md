# Integration 10: Quantization Config Comparison

## BitsAndBytesConfig Parameters

### load_in_4bit vs load_in_8bit

| Parameter | 4-bit | 8-bit | Recommendation |
|-----------|-------|-------|----------------|
| **Memory reduction** | 93.75% | 87.5% | 4-bit for max savings |
| **Quality loss** | ~5% | ~1% | 8-bit for critical tasks |
| **Speed** | Fast | Faster | 8-bit slightly faster |

---

### bnb_4bit_quant_type

| Type | Distribution | Quality | Use Case |
|------|--------------|---------|----------|
| **nf4** | Normal | Best | Pre-trained models |
| **fp4** | Uniform | Good | General use |

```python
bnb_4bit_quant_type="nf4"  # Recommended for LLMs
```

---

### bnb_4bit_compute_dtype

| Dtype | Speed | Stability | VRAM |
|-------|-------|-----------|------|
| **float16** | Fast | Good | Low |
| **bfloat16** | Fast | Better | Low |
| **float32** | Slow | Best | High |

```python
bnb_4bit_compute_dtype=torch.bfloat16  # Best balance
```

---

### bnb_4bit_use_double_quant

```python
bnb_4bit_use_double_quant=True  # +0.4 bits savings
```

**Trade-off:** Minimal quality loss, ~3% more memory savings

---

## Optimal Configs

### For Inference

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,  # Max savings
)
```

---

### For Fine-tuning (QLoRA)

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,  # Better quality
)
```

---

### For Critical Tasks

```python
BitsAndBytesConfig(
    load_in_8bit=True,  # Less aggressive
)
```

---

Next: **22_Quantization_Memory_Analysis.md**
