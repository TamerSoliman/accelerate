# Integration 18: Sharded Checkpoint Tutorial

## Overview

Large models split checkpoints across multiple files (shards) to avoid single large file limits.

---

## Checkpoint Formats

### 1. Single File

```
pytorch_model.bin  (28GB for 7B model)
```

**Problem:** Slow to load, exceeds some filesystem limits

---

### 2. Sharded Checkpoint

```
pytorch_model-00001-of-00015.bin  (2GB)
pytorch_model-00002-of-00015.bin  (2GB)
...
pytorch_model-00015-of-00015.bin  (2GB)
pytorch_model.bin.index.json      (metadata)
```

**Benefits:**
- Parallel loading
- Easier to handle
- Can load individual shards

---

## Loading Sharded Checkpoints

```python
from accelerate import load_checkpoint_in_model

# Automatically handles sharded checkpoints
model = load_checkpoint_in_model(
    model,
    checkpoint="./model",  # Directory with .index.json
    device_map=device_map,
)
```

---

## Index JSON Structure

```json
{
  "metadata": {
    "total_size": 28000000000
  },
  "weight_map": {
    "model.embed_tokens.weight": "pytorch_model-00001-of-00015.bin",
    "model.layers.0.self_attn.q_proj.weight": "pytorch_model-00001-of-00015.bin",
    "model.layers.10.self_attn.q_proj.weight": "pytorch_model-00002-of-00015.bin",
    ...
  }
}
```

---

## Creating Sharded Checkpoints

```python
from accelerate import save_state_dict_sharded

save_state_dict_sharded(
    model.state_dict(),
    save_directory="./model",
    max_shard_size="2GB",  # Max shard size
)
```

---

Next: **30_Tied_Parameters_Guide.md**
