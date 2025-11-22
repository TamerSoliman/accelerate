# Integration 11: Quantization Memory Analysis

## Memory Breakdown by Model Size

### 7B Parameters

| Precision | Model | Activations* | Total | GPU Fit |
|-----------|-------|--------------|-------|---------|
| FP32      | 28GB  | 2GB          | 30GB  | A100 40GB |
| FP16      | 14GB  | 1GB          | 15GB  | RTX 3090 24GB |
| 8-bit     | 7GB   | 1GB          | 8GB   | RTX 3090 24GB |
| 4-bit     | 3.5GB | 1GB          | 4.5GB | RTX 3080 10GB |

*Batch size = 1

---

### 13B Parameters

| Precision | Model | Total | GPU Fit |
|-----------|-------|-------|---------|
| FP16      | 26GB  | 27GB  | A100 40GB |
| 8-bit     | 13GB  | 14GB  | RTX 3090 24GB |
| 4-bit     | 6.5GB | 7.5GB | RTX 3090 24GB |

---

### 70B Parameters

| Precision | Model | Total | Setup |
|-----------|-------|-------|-------|
| FP16      | 140GB | 141GB | 4× A100 40GB |
| 8-bit     | 70GB  | 71GB  | 2× A100 40GB |
| 4-bit     | 35GB  | 36GB  | 1× A100 40GB |

---

## Activation Memory

Scales with:
- Batch size
- Sequence length
- Hidden size

**Formula:**
```
activation_memory ≈ batch_size × seq_len × hidden_size × layers × 2 bytes (FP16)
```

---

## Optimizer Memory (Training)

| Optimizer | Multiplier | Example (7B FP16) |
|-----------|------------|-------------------|
| SGD       | 0× | 0GB |
| AdamW     | 2× | 28GB |

**QLoRA advantage:** Only trains adapters (~1% params), minimal optimizer memory

---

## Real-World Examples

### Llama-2-70B on Single GPU

```python
# Requires: A100 80GB or H100
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    load_in_4bit=True,  # 140GB → 35GB
    device_map="auto",
)
```

---

### Llama-2-70B on Consumer Hardware

```python
# RTX 3090 24GB × 2
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    load_in_4bit=True,
    device_map="auto",
    max_memory={0: "22GB", 1: "22GB"},
)
```

---

Next: **23_PEFT_LoRA_Integration.md**
