# Integration 19: Tied Parameters Guide

## Overview

Tied parameters = shared weights (e.g., input embeddings = output embeddings). Must stay on same device.

**File:** `src/accelerate/utils/modeling.py` (find_tied_parameters, retie_parameters)

---

## Common Tied Parameters

### Language Models

```python
# Input and output embeddings often tied
model.embed_tokens.weight is model.lm_head.weight  # True in many LMs
```

**Why:** Reduces parameters, improves training

---

## Detection

```python
from accelerate.utils import find_tied_parameters

tied_params = find_tied_parameters(model)
print(tied_params)
# [['model.embed_tokens.weight', 'lm_head.weight']]
```

---

## Device Map Constraints

```python
# ❌ BAD: Tied params on different devices
device_map = {
    "model.embed_tokens": 0,   # GPU 0
    "lm_head": 1,               # GPU 1 - ERROR!
}

# ✅ GOOD: Tied params on same device
device_map = {
    "model.embed_tokens": 0,
    "lm_head": 0,  # Same device
}
```

---

## Retying After Load

```python
from accelerate.utils import retie_parameters

# After loading checkpoint, retie if needed
retie_parameters(model, tied_params)
```

---

Next: **31_Memory_Efficient_Loading.md**
