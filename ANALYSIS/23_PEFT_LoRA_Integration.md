# Integration 12: PEFT/LoRA Integration

## Overview

PEFT (Parameter-Efficient Fine-Tuning) + BitsAndBytes enables fine-tuning massive models on consumer GPUs.

---

## QLoRA Pattern

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Step 1: Load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Step 2: Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Step 3: Add LoRA adapters
lora_config = LoraConfig(
    r=16,                                # Rank
    lora_alpha=32,                       # Scaling factor
    target_modules=["q_proj", "v_proj"], # Which layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Step 4: Train (only LoRA parameters updated)
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        output_dir="./lora_output",
    ),
)

trainer.train()

# Step 5: Save adapters (tiny, ~50MB)
model.save_pretrained("./lora_adapters")
```

---

## LoRA Configuration

### Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| **r** | Rank (matrix size) | 8-64 |
| **lora_alpha** | Scaling factor | 2× rank |
| **target_modules** | Which layers | q_proj, v_proj |
| **lora_dropout** | Dropout rate | 0.05-0.1 |

---

### Target Modules by Model

**Llama:**
```python
target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
```

**GPT-2:**
```python
target_modules=["c_attn", "c_proj"]
```

**BLOOM:**
```python
target_modules=["query_key_value", "dense"]
```

---

## Memory Comparison

### Llama-2-7B Fine-tuning

| Method | Memory | Trainable Params | Time |
|--------|--------|------------------|------|
| Full FP16 | ~40GB | 7B (100%) | 1× |
| LoRA FP16 | ~20GB | 4.2M (0.06%) | 1.2× |
| QLoRA 4-bit | ~9GB | 4.2M (0.06%) | 1.5× |

---

### Llama-2-70B Fine-tuning

| Method | GPU Required | Memory |
|--------|--------------|--------|
| Full FP16 | 8× A100 80GB | ~560GB |
| QLoRA 4-bit | 1× A100 80GB | ~48GB |

---

## Loading Fine-tuned Model

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map="auto",
)

# Load adapters
model = PeftModel.from_pretrained(base_model, "./lora_adapters")

# Inference
outputs = model.generate(**inputs)
```

---

## File References

- BitsAndBytes integration: `src/accelerate/utils/bnb.py`
- PEFT (external library): `pip install peft`

---

Next: **Group 4 - Hook System**
