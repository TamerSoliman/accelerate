# Integration 4: Multi-Tracker Best Practices

## Overview

Accelerate allows logging to **multiple trackers simultaneously** with a single `log()` call. This enables powerful combinations like local TensorBoard for debugging + cloud WandB for collaboration.

**Key Benefit:** Get the advantages of multiple platforms without duplicating logging code.

---

## Quick Start: Using Multiple Trackers

```python
from accelerate import Accelerator

# Log to both TensorBoard AND WandB
accelerator = Accelerator(
    log_with=["tensorboard", "wandb"],
    project_dir="./logs",  # For TensorBoard
)

accelerator.init_trackers(
    project_name="multi_tracker_experiment",
    config={"learning_rate": 1e-4, "batch_size": 32},
    init_kwargs={
        "tensorboard": {"flush_secs": 60},
        "wandb": {"entity": "my_team", "tags": ["experiment_1"]},
    },
)

# Single call logs to BOTH trackers
for step, batch in enumerate(train_dataloader):
    loss = train_step(batch)
    accelerator.log({"train/loss": loss}, step=step)

# Cleanup both trackers
accelerator.end_training()
```

---

## Recommended Combinations

### 1. Local + Cloud (Best for Most Use Cases)

```python
accelerator = Accelerator(
    log_with=["tensorboard", "wandb"],
    project_dir="./logs",
)
```

**Benefits:**
- **TensorBoard**: Fast local debugging, offline access
- **WandB**: Cloud collaboration, advanced visualizations, model registry

**Use case:** Development + team sharing

---

### 2. Production Tracking + Experiment Management

```python
accelerator = Accelerator(
    log_with=["mlflow", "wandb"],
)
```

**Benefits:**
- **MLflow**: Model registry, deployment, production lifecycle
- **WandB**: Rich visualizations, hyperparameter sweeps

**Use case:** Production ML pipelines with experiment tracking

---

### 3. Local Redundancy

```python
accelerator = Accelerator(
    log_with=["tensorboard", "aim"],
    project_dir="./logs",
)
```

**Benefits:**
- **TensorBoard**: Standard PyTorch logging
- **Aim**: Powerful querying, fast local storage

**Use case:** Research without cloud dependency

---

### 4. Triple Tracking (Maximum Coverage)

```python
accelerator = Accelerator(
    log_with=["tensorboard", "wandb", "mlflow"],
    project_dir="./logs",
)
```

**Benefits:**
- **TensorBoard**: Local debugging
- **WandB**: Collaboration and visualization
- **MLflow**: Model registry and deployment

**Use case:** Enterprise ML with full lifecycle tracking

---

## Tracker-Specific Configuration

### Per-Tracker Initialization

```python
accelerator.init_trackers(
    project_name="my_project",
    config={
        "learning_rate": 1e-4,
        "batch_size": 32,
        "model": "bert-base",
    },
    init_kwargs={
        "tensorboard": {
            "flush_secs": 60,  # Flush to disk every 60s
            "max_queue": 100,
        },
        "wandb": {
            "entity": "my_team",
            "tags": ["bert", "baseline"],
            "group": "experiment_1",
            "job_type": "train",
            "notes": "Testing new hyperparameters",
        },
        "mlflow": {
            "run_name": "run_001",
            "tags": {"version": "v1.0", "dataset": "imdb"},
            "description": "Baseline BERT model",
        },
    },
)
```

---

### Per-Tracker Logging Parameters

```python
accelerator.log(
    {"train/loss": loss, "train/accuracy": acc},
    step=step,
    log_kwargs={
        "wandb": {
            "commit": False,  # Don't increment step in WandB
        },
        "tensorboard": {
            "walltime": time.time(),  # Custom timestamp
        },
    },
)
```

---

## Advanced Patterns

### Pattern 1: Tracker-Specific Data

Some data types are only supported by specific trackers:

```python
# Log scalar metrics to ALL trackers
accelerator.log({"train/loss": loss}, step=step)

# Log images only to WandB (TensorBoard doesn't support via accelerate.log)
wandb_tracker = accelerator.get_tracker("wandb")
if wandb_tracker:
    wandb_tracker.log_images({"samples": images}, step=step)

# Log model graph only to TensorBoard
tb_tracker = accelerator.get_tracker("tensorboard")
if tb_tracker:
    tb_tracker.writer.add_graph(model, input_sample)

# Log artifacts only to MLflow
mlflow_tracker = accelerator.get_tracker("mlflow", unwrap=True)
if mlflow_tracker:
    import mlflow
    mlflow.log_artifact("model.pt")
```

---

### Pattern 2: Conditional Tracker Access

```python
def log_advanced_metrics(accelerator, metrics, step):
    # Log basic metrics to all trackers
    accelerator.log(metrics, step=step)

    # Log tracker-specific data
    try:
        wandb_tracker = accelerator.get_tracker("wandb")
        # WandB-specific logging (tables, etc.)
        wandb_tracker.log_table("predictions", columns, data, step=step)
    except ValueError:
        pass  # WandB not enabled

    try:
        tb_tracker = accelerator.get_tracker("tensorboard")
        # TensorBoard-specific logging (histograms, etc.)
        for name, param in model.named_parameters():
            tb_tracker.writer.add_histogram(f"params/{name}", param, step)
    except ValueError:
        pass  # TensorBoard not enabled
```

---

### Pattern 3: Metric Namespacing

Use prefixes to organize metrics across trackers:

```python
# Clear organization
accelerator.log({
    # Training metrics
    "train/loss": train_loss,
    "train/accuracy": train_acc,
    "train/learning_rate": lr,

    # Validation metrics
    "val/loss": val_loss,
    "val/accuracy": val_acc,

    # System metrics
    "system/gpu_memory_mb": gpu_memory,
    "system/samples_per_sec": throughput,

    # Model metrics
    "model/grad_norm": grad_norm,
    "model/weight_norm": weight_norm,
}, step=step)
```

**Benefit:** Easy filtering in TensorBoard/WandB UI

---

## Performance Considerations

### 1. Logging Frequency

Logging to multiple trackers increases overhead:

```python
# ❌ BAD: Log every iteration to all trackers
for step, batch in enumerate(train_dataloader):
    loss = train_step(batch)
    accelerator.log({"train/loss": loss}, step=step)  # Slow with multiple trackers

# ✅ GOOD: Reduce logging frequency
for step, batch in enumerate(train_dataloader):
    loss = train_step(batch)

    if step % 10 == 0:  # Log every 10 steps
        accelerator.log({"train/loss": loss}, step=step)

    if step % 100 == 0:  # Detailed metrics every 100 steps
        accelerator.log({
            "train/loss": loss,
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/grad_norm": grad_norm,
        }, step=step)
```

---

### 2. Async Logging (Advanced)

For minimal training overhead:

```python
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

def async_log(accelerator, values, step):
    # Runs in background thread
    accelerator.log(values, step=step)

# In training loop
for step, batch in enumerate(train_dataloader):
    loss = train_step(batch)

    if step % 10 == 0:
        # Submit logging to background thread
        executor.submit(async_log, accelerator, {"train/loss": loss.item()}, step)

# Wait for all logging to complete before exit
executor.shutdown(wait=True)
accelerator.end_training()
```

---

### 3. Selective Tracker Logging

Log expensive data to only one tracker:

```python
# Log scalar metrics to ALL trackers (fast)
accelerator.log({"train/loss": loss}, step=step)

# Log images only to WandB (avoid TensorBoard disk I/O)
if step % 500 == 0:
    wandb_tracker = accelerator.get_tracker("wandb")
    if wandb_tracker:
        wandb_tracker.log_images({"samples": generated_images}, step=step)
```

---

## Cost Optimization

### Strategy 1: Free Tier Combinations

```python
# All free, no limits
accelerator = Accelerator(
    log_with=["tensorboard", "mlflow"],  # Both open-source
    project_dir="./logs",
)
```

---

### Strategy 2: Minimal Cloud Usage

```python
# Primary: TensorBoard (free, local)
# Backup: WandB (free tier: 100GB)
accelerator = Accelerator(
    log_with=["tensorboard", "wandb"],
    project_dir="./logs",
)

# Log all metrics to TensorBoard, only summaries to WandB
for step, batch in enumerate(train_dataloader):
    loss = train_step(batch)

    # TensorBoard: log every iteration
    tb_tracker = accelerator.get_tracker("tensorboard")
    tb_tracker.log({"train/loss": loss}, step=step)

    # WandB: log every 100 iterations
    if step % 100 == 0:
        wandb_tracker = accelerator.get_tracker("wandb")
        wandb_tracker.log({"train/loss": loss}, step=step)
```

---

## Distributed Training

### Single Tracker Behavior

In distributed training (multi-GPU), logging happens **only on rank 0** by default:

```python
# 4 GPU training
accelerator = Accelerator(log_with="tensorboard")
accelerator.init_trackers("my_project")

# This logs ONCE, even with 4 GPUs
accelerator.log({"train/loss": loss}, step=step)
```

---

### Multi-Tracker Behavior

Same behavior applies to multiple trackers:

```python
# 4 GPU training
accelerator = Accelerator(log_with=["tensorboard", "wandb"])
accelerator.init_trackers("my_project")

# Logs ONCE to TensorBoard AND ONCE to WandB (from rank 0 only)
accelerator.log({"train/loss": loss}, step=step)
```

---

### WandB Group Runs

WandB can create separate runs for each process:

```python
import os

accelerator = Accelerator(log_with="wandb")

# Each GPU gets its own run, but grouped together
accelerator.init_trackers(
    "my_project",
    init_kwargs={
        "wandb": {
            "group": "experiment_1",  # Group all runs
            "name": f"gpu_{os.environ.get('RANK', 0)}",  # Unique name per GPU
        }
    },
)
```

**Result:** WandB UI shows all GPU runs grouped together.

---

## Error Handling

### Graceful Tracker Failures

```python
# If one tracker fails, others continue
try:
    accelerator = Accelerator(log_with=["tensorboard", "wandb", "mlflow"])
    accelerator.init_trackers("my_project")
except Exception as e:
    print(f"Tracker initialization failed: {e}")
    # Fall back to single tracker
    accelerator = Accelerator(log_with="tensorboard", project_dir="./logs")
    accelerator.init_trackers("my_project")
```

---

### Tracker Availability Check

```python
from accelerate.tracking import get_available_trackers

available = get_available_trackers()
print(f"Available trackers: {available}")

# Use all available trackers
accelerator = Accelerator(log_with=available)
```

---

## Migration to Multi-Tracker

### Step 1: Add Second Tracker (No Code Changes)

```python
# Original (single tracker)
accelerator = Accelerator(log_with="tensorboard", project_dir="./logs")

# Updated (multi-tracker)
accelerator = Accelerator(
    log_with=["tensorboard", "wandb"],  # Just add to list
    project_dir="./logs",
)

# NO CHANGES to logging code needed!
accelerator.log({"train/loss": loss}, step=step)
```

---

### Step 2: Add Tracker-Specific Features (Optional)

```python
# After migration, optionally add tracker-specific features
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
wandb_tracker.watch(model)  # Log gradients (WandB-only feature)
```

---

## Complete Multi-Tracker Example

```python
from accelerate import Accelerator
from transformers import AutoModel, AdamW
from torch.utils.data import DataLoader

def train():
    # Multi-tracker setup
    accelerator = Accelerator(
        log_with=["tensorboard", "wandb", "mlflow"],
        project_dir="./outputs",
        mixed_precision="fp16",
    )

    model = AutoModel.from_pretrained("bert-base-uncased")
    optimizer = AdamW(model.parameters(), lr=1e-4)
    train_dataloader = DataLoader(dataset, batch_size=32)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Initialize all trackers with specific configs
    hparams = {
        "model": "bert-base-uncased",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "num_epochs": 3,
    }

    accelerator.init_trackers(
        project_name="bert_finetuning",
        config=hparams,
        init_kwargs={
            "tensorboard": {"flush_secs": 60},
            "wandb": {"entity": "my_team", "tags": ["bert", "v1"]},
            "mlflow": {"run_name": "run_001", "tags": {"version": "v1.0"}},
        },
    )

    # Training loop - single log call logs to ALL trackers
    global_step = 0
    for epoch in range(3):
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # Log every 10 steps to all trackers
            if global_step % 10 == 0:
                accelerator.log({
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch,
                }, step=global_step)

            global_step += 1

        # Validation
        val_loss = validate(model, val_dataloader)

        # Log validation metrics to all trackers
        accelerator.log({"val/loss": val_loss}, step=epoch)

        # Tracker-specific logging
        # WandB: Log model predictions table
        wandb_tracker = accelerator.get_tracker("wandb")
        if wandb_tracker:
            predictions = generate_predictions(model, val_dataloader)
            wandb_tracker.log_table("predictions", columns, predictions, step=epoch)

        # MLflow: Log model checkpoint
        if epoch == 2:  # Final epoch
            accelerator.save_state("final_model")
            mlflow_tracker = accelerator.get_tracker("mlflow", unwrap=True)
            if mlflow_tracker:
                import mlflow.pytorch
                mlflow.pytorch.log_model(model, "model")

    # Cleanup all trackers
    accelerator.end_training()

if __name__ == "__main__":
    train()
```

---

## Summary

### Key Benefits of Multi-Tracking

1. **Redundancy**: Logs saved to multiple locations
2. **Best of Both Worlds**: Local debugging + cloud collaboration
3. **No Code Duplication**: Single `log()` call logs to all trackers
4. **Flexibility**: Easy to add/remove trackers without code changes

### Recommended Combinations

| Use Case | Combination | Why |
|----------|-------------|-----|
| Development | TensorBoard + WandB | Local debug + cloud share |
| Production | MLflow + WandB | Model registry + visualization |
| Research | TensorBoard + Aim | Fast local tracking with querying |
| Enterprise | TensorBoard + WandB + MLflow | Complete coverage |
| Open Source | TensorBoard + MLflow | No cloud dependency |

### Performance Tips

1. **Reduce logging frequency** for multiple trackers
2. **Use async logging** for expensive operations
3. **Log images/videos only to one tracker** (avoid redundancy)
4. **Check tracker availability** before use

### File References

- **Multi-tracker init**: `src/accelerate/accelerator.py:3190-3209`
- **Multi-tracker log**: `src/accelerate/accelerator.py:3270-3271`
- **Tracker filtering**: `src/accelerate/tracking.py:1271-1327`

---

## Next Steps

This completes the Experiment Tracking integration guides. Next sections cover:
- **Group 2**: Large Model Handling (big_modeling.py)
- **Group 3**: BitsAndBytes Quantization (bnb.py)
- **Group 4**: Hook System (hooks.py)
- **Group 5**: Model Utilities (modeling.py)
