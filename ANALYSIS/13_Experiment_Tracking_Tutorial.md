# Integration 2: Experiment Tracking Tutorial

## Overview

This tutorial shows how to use Accelerate's unified experiment tracking API to log metrics to TensorBoard, Weights & Biases, MLflow, and other platforms with minimal code changes.

**Key Benefit:** Switch between 9 different tracking platforms by changing a single parameter, with no changes to your training code.

---

## Quick Start

### Minimal Example

```python
from accelerate import Accelerator

# Step 1: Specify tracker in Accelerator init
accelerator = Accelerator(log_with="tensorboard")

# Step 2: Initialize tracker with project name
accelerator.init_trackers(project_name="my_experiment")

# Step 3: Log metrics during training
for step, batch in enumerate(train_dataloader):
    loss = train_step(batch)
    accelerator.log({"train/loss": loss.item()}, step=step)

# Step 4: Clean up at the end
accelerator.end_training()
```

**That's it!** Change `log_with="tensorboard"` to `log_with="wandb"` to switch to Weights & Biases.

---

## Complete Training Example

### Full Training Script with Tracking

```python
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW
import torch

def train():
    # ========== SETUP ==========
    # Initialize Accelerator with tracker
    accelerator = Accelerator(
        log_with="tensorboard",           # Can be "wandb", "mlflow", "comet_ml", etc.
        project_dir="./outputs",          # Where to save logs (for TensorBoard/Aim)
        mixed_precision="fp16",
    )

    # Load model, optimizer, dataloader
    model = AutoModel.from_pretrained("bert-base-uncased")
    optimizer = AdamW(model.parameters(), lr=1e-4)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Prepare for distributed training
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # ========== TRACKING INITIALIZATION ==========
    # Define hyperparameters to log
    hparams = {
        "model_name": "bert-base-uncased",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "num_epochs": 3,
        "mixed_precision": "fp16",
        "num_gpus": accelerator.num_processes,
    }

    # Initialize tracker and log hyperparameters
    accelerator.init_trackers(
        project_name="bert_finetuning",
        config=hparams,  # Logs hyperparameters
    )

    # ========== TRAINING LOOP ==========
    global_step = 0

    for epoch in range(3):
        model.train()
        epoch_loss = 0

        for batch in train_dataloader:
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # ===== LOG TRAINING METRICS =====
            accelerator.log({
                "train/loss": loss.item(),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
            }, step=global_step)

            epoch_loss += loss.item()
            global_step += 1

        # ===== LOG EPOCH METRICS =====
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        accelerator.log({
            "epoch/avg_loss": avg_epoch_loss,
        }, step=epoch)

        # Validation
        if epoch % 1 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(val_dataloader)

            # ===== LOG VALIDATION METRICS =====
            accelerator.log({
                "val/loss": avg_val_loss,
            }, step=epoch)

            accelerator.print(f"Epoch {epoch}: train_loss={avg_epoch_loss:.4f}, val_loss={avg_val_loss:.4f}")

    # ========== CLEANUP ==========
    accelerator.end_training()

if __name__ == "__main__":
    train()
```

**Output (TensorBoard):**
```
TensorBoard logs saved to: ./outputs/bert_finetuning/
Run: tensorboard --logdir=./outputs/bert_finetuning
```

---

## Switching Between Trackers

### Using TensorBoard

```python
accelerator = Accelerator(
    log_with="tensorboard",
    project_dir="./logs",  # Required for TensorBoard
)
accelerator.init_trackers("my_project")
```

**View logs:**
```bash
tensorboard --logdir=./logs/my_project
```

---

### Using Weights & Biases

```python
accelerator = Accelerator(
    log_with="wandb",
    # No project_dir needed - WandB is cloud-based
)

accelerator.init_trackers(
    project_name="my_project",
    init_kwargs={
        "wandb": {
            "entity": "my_username",  # Your WandB username/team
            "tags": ["experiment_1", "bert"],
            "notes": "Testing new hyperparameters",
        }
    },
)
```

**View logs:** https://wandb.ai/my_username/my_project

---

### Using MLflow

```python
accelerator = Accelerator(log_with="mlflow")

accelerator.init_trackers(
    project_name="my_experiment",
    init_kwargs={
        "mlflow": {
            "run_name": "run_001",
            "tags": {"version": "v1.0", "dataset": "imdb"},
            "description": "Baseline BERT model",
        }
    },
)
```

**View logs:**
```bash
mlflow ui  # Opens at http://localhost:5000
```

---

### Using Multiple Trackers Simultaneously

**Log to TensorBoard AND WandB at the same time:**

```python
accelerator = Accelerator(
    log_with=["tensorboard", "wandb"],
    project_dir="./logs",  # For TensorBoard
)

accelerator.init_trackers(
    project_name="multi_tracker_experiment",
    config={
        "learning_rate": 1e-4,
        "batch_size": 32,
    },
    init_kwargs={
        "tensorboard": {
            "flush_secs": 60,  # Flush to disk every 60 seconds
        },
        "wandb": {
            "entity": "my_username",
            "tags": ["multi_tracker"],
        },
    },
)

# Single log call logs to BOTH trackers
accelerator.log({"train/loss": 0.5}, step=100)
```

**Benefit:** Local TensorBoard logs for debugging + cloud WandB logs for sharing.

---

## Logging Different Data Types

### 1. Scalar Metrics (Loss, Accuracy, etc.)

```python
accelerator.log({
    "train/loss": 0.342,
    "train/accuracy": 0.87,
    "train/f1_score": 0.85,
}, step=100)
```

---

### 2. Multiple Related Metrics (Per-Class Accuracy)

```python
# TensorBoard supports nested dicts for grouped scalars
accelerator.log({
    "class_accuracy": {
        "class_0": 0.92,
        "class_1": 0.88,
        "class_2": 0.90,
    }
}, step=100)
```

---

### 3. Text Logging (Model Predictions)

```python
# Only supported by TensorBoard, WandB, CometML, Aim
tensorboard_tracker = accelerator.get_tracker("tensorboard")
tensorboard_tracker.writer.add_text("predictions", "Sample prediction: positive", step=100)
```

---

### 4. Images (Attention Maps, Generated Images)

```python
# Get the specific tracker for image logging
tracker = accelerator.get_tracker("tensorboard")

# TensorBoard image logging
import torchvision
img_grid = torchvision.utils.make_grid(images)
tracker.writer.add_image("generated_images", img_grid, step=100)

# WandB image logging
wandb_tracker = accelerator.get_tracker("wandb")
import wandb
wandb_tracker.log_images({"samples": [wandb.Image(img) for img in images]}, step=100)
```

---

### 5. Hyperparameter Sweeps

**WandB Sweep Example:**

```python
import wandb

# Define sweep config
sweep_config = {
    "method": "bayes",
    "parameters": {
        "learning_rate": {"min": 1e-5, "max": 1e-3},
        "batch_size": {"values": [16, 32, 64]},
    },
}

sweep_id = wandb.sweep(sweep_config, project="my_project")

def train_with_sweep():
    # Get sweep config from WandB
    wandb.init()
    config = wandb.config

    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers("my_project")

    # Use config values
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)

    # Train and log...

wandb.agent(sweep_id, function=train_with_sweep)
```

---

## Advanced Usage

### 1. Accessing Native Tracker APIs

Sometimes you need tracker-specific features. Use `get_tracker()`:

```python
accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers("my_project")

# Get the native WandB run object
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)

# Now use native WandB API
wandb_tracker.watch(model)  # Log gradients
wandb_tracker.save("model.pt")  # Save artifact
wandb_tracker.alert(title="Training complete!", text="Model achieved 95% accuracy")
```

**Common use cases:**
- **WandB:** `wandb_tracker.watch(model)` for gradient logging
- **TensorBoard:** `tb_tracker.add_graph(model, input_sample)` for model graph
- **MLflow:** `mlflow_tracker.log_artifact("model.pt")` for artifact logging
- **CometML:** `comet_tracker.log_model(name, file_or_folder)` for model versioning

---

### 2. Per-Tracker Logging Parameters

Pass tracker-specific kwargs via `log_kwargs`:

```python
accelerator.log(
    {"train/loss": 0.5},
    step=100,
    log_kwargs={
        "wandb": {"commit": False},  # Don't create new step in WandB
        "tensorboard": {"walltime": time.time()},  # Custom timestamp
    },
)
```

---

### 3. Conditional Logging (Main Process Only)

Accelerate automatically logs only on the main process, but you can check manually:

```python
if accelerator.is_main_process:
    # Custom logging logic
    with open("custom_log.txt", "a") as f:
        f.write(f"Step {step}: loss={loss}\n")

# This automatically runs only on main process
accelerator.log({"train/loss": loss}, step=step)
```

---

### 4. Logging Frequency Control

**Don't log every step (too much data):**

```python
for step, batch in enumerate(train_dataloader):
    loss = train_step(batch)

    # Log every 10 steps
    if step % 10 == 0:
        accelerator.log({"train/loss": loss.item()}, step=step)

    # Log every 100 steps with more metrics
    if step % 100 == 0:
        accelerator.log({
            "train/loss": loss.item(),
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/grad_norm": grad_norm,
        }, step=step)
```

---

### 5. Offline Mode (WandB/MLflow)

**WandB offline mode** (logs locally, syncs later):

```python
import os
os.environ["WANDB_MODE"] = "offline"

accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers("my_project")

# Train normally...

# Later, sync to cloud
# wandb sync ./wandb/offline-run-xxx
```

**MLflow local mode** (no tracking server):

```python
accelerator = Accelerator(log_with="mlflow")
accelerator.init_trackers(
    "my_project",
    init_kwargs={
        "mlflow": {
            "logging_dir": "./mlruns",  # Local directory
        }
    },
)
```

---

## Distributed Training Considerations

### 1. Automatic Main Process Logging

In multi-GPU training, only process 0 logs to avoid duplicates:

```python
# 4 GPU training
accelerator = Accelerator(log_with="tensorboard")
accelerator.init_trackers("my_project")

# This logs only once, even with 4 GPUs
accelerator.log({"train/loss": loss}, step=step)
```

**Behind the scenes:**
- `@on_main_process` decorator checks `accelerator.is_main_process`
- Only process with `rank == 0` executes logging calls
- Other processes skip logging silently

---

### 2. Gathering Metrics from All Processes

If you need to log metrics from all GPUs:

```python
# Each GPU computes its own accuracy
local_accuracy = compute_accuracy(predictions, labels)

# Gather from all GPUs
all_accuracies = accelerator.gather(local_accuracy)

# Main process logs average
if accelerator.is_main_process:
    avg_accuracy = all_accuracies.mean().item()
    accelerator.log({"val/accuracy": avg_accuracy}, step=epoch)
```

---

### 3. WandB Distributed Logging

WandB handles distributed training differently:

```python
# WandB creates separate runs for each process by default
# To log from all processes to ONE run, use:
accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(
    "my_project",
    init_kwargs={
        "wandb": {
            "group": "experiment_1",  # Groups runs together
            "job_type": "train",
        }
    },
)
```

**Result:** Each GPU gets its own run, but grouped in WandB UI.

---

## Tracking Lifecycle

### Complete Workflow

```python
from accelerate import Accelerator

# 1. INITIALIZATION
accelerator = Accelerator(
    log_with=["tensorboard", "wandb"],
    project_dir="./outputs",
)

# 2. TRACKER SETUP (after prepare(), before training)
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

accelerator.init_trackers(
    project_name="my_project",
    config={
        "learning_rate": 1e-4,
        "batch_size": 32,
    },
)

# 3. TRAINING LOOP (log metrics)
for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    accelerator.log({"train/loss": loss.item()}, step=step)

# 4. VALIDATION (log validation metrics)
val_loss = validate(model, val_dataloader)
accelerator.log({"val/loss": val_loss}, step=step)

# 5. CLEANUP (flush logs, mark run complete)
accelerator.end_training()
```

**Timing:**
1. **Before training:** `init_trackers()` + `store_init_configuration()`
2. **During training:** `log()` for metrics
3. **After training:** `end_training()` calls `tracker.finish()`

---

## Common Patterns

### Pattern 1: Best Model Checkpointing

```python
best_loss = float("inf")

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_dataloader)
    val_loss = validate(model, val_dataloader)

    # Log metrics
    accelerator.log({
        "train/loss": train_loss,
        "val/loss": val_loss,
    }, step=epoch)

    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        accelerator.wait_for_everyone()
        accelerator.save_state("checkpoints/best_model")

        # Log best model indicator
        accelerator.log({"best_val_loss": best_loss}, step=epoch)
```

---

### Pattern 2: Learning Rate Schedules

```python
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=1000
)

for step, batch in enumerate(train_dataloader):
    loss = train_step(batch)
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()  # Update learning rate
    optimizer.zero_grad()

    # Log learning rate along with loss
    accelerator.log({
        "train/loss": loss.item(),
        "train/lr": scheduler.get_last_lr()[0],
    }, step=step)
```

---

### Pattern 3: Gradient Monitoring

```python
for step, batch in enumerate(train_dataloader):
    loss = train_step(batch)
    accelerator.backward(loss)

    # Compute gradient norm
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    # Log gradient norm
    accelerator.log({
        "train/loss": loss.item(),
        "train/grad_norm": total_norm,
    }, step=step)

    optimizer.step()
    optimizer.zero_grad()
```

---

### Pattern 4: Multi-Phase Training

```python
# Phase 1: Pretrain
accelerator.init_trackers("pretraining", config={"phase": "pretrain"})
for epoch in range(pretrain_epochs):
    loss = pretrain_epoch(model, pretrain_dataloader)
    accelerator.log({"pretrain/loss": loss}, step=epoch)
accelerator.end_training()

# Phase 2: Fine-tune
accelerator.init_trackers("finetuning", config={"phase": "finetune"})
for epoch in range(finetune_epochs):
    loss = finetune_epoch(model, finetune_dataloader)
    accelerator.log({"finetune/loss": loss}, step=epoch)
accelerator.end_training()
```

---

## Tracker-Specific Features

### TensorBoard

**Model Graph Visualization:**
```python
tb_tracker = accelerator.get_tracker("tensorboard")
dummy_input = torch.randn(1, 3, 224, 224).to(accelerator.device)
tb_tracker.writer.add_graph(model, dummy_input)
```

**Embedding Visualization (t-SNE):**
```python
embeddings = model.get_embeddings(data)
metadata = [label for label in labels]
tb_tracker.writer.add_embedding(embeddings, metadata=metadata, label_img=images)
```

---

### Weights & Biases

**Watch Model (Log Gradients/Parameters):**
```python
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
wandb_tracker.watch(model, log="all", log_freq=100)
```

**Log Tables (Predictions):**
```python
columns = ["image", "prediction", "ground_truth"]
data = [[wandb.Image(img), pred, gt] for img, pred, gt in zip(images, preds, labels)]

wandb_tracker = accelerator.get_tracker("wandb")
wandb_tracker.log_table("predictions", columns=columns, data=data, step=epoch)
```

**Save Model as Artifact:**
```python
accelerator.save_state("final_model")

wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
artifact = wandb.Artifact("bert-finetuned", type="model")
artifact.add_dir("final_model")
wandb_tracker.log_artifact(artifact)
```

---

### MLflow

**Log Model with Signature:**
```python
import mlflow.pytorch

mlflow_tracker = accelerator.get_tracker("mlflow", unwrap=True)

# Log PyTorch model
mlflow.pytorch.log_model(
    model,
    "model",
    registered_model_name="bert-classifier"
)
```

**Log Artifacts (Plots, Data):**
```python
# Save plot
import matplotlib.pyplot as plt
plt.plot(losses)
plt.savefig("loss_curve.png")

mlflow_tracker = accelerator.get_tracker("mlflow", unwrap=True)
mlflow_tracker.log_artifact("loss_curve.png")
```

---

## Troubleshooting

### Issue 1: Tracker Not Available

**Error:**
```
Tried adding logger wandb, but package is unavailable in the system.
```

**Solution:** Install the tracker library:
```bash
pip install wandb  # For WandB
pip install tensorboard  # For TensorBoard
pip install mlflow  # For MLflow
```

---

### Issue 2: TensorBoard Requires Logging Directory

**Error:**
```
ValueError: Logging with `tensorboard` requires a `logging_dir` to be passed in.
```

**Solution:** Specify `project_dir` in Accelerator:
```python
accelerator = Accelerator(
    log_with="tensorboard",
    project_dir="./outputs",  # Required for TensorBoard/Aim
)
```

---

### Issue 3: Multiple Processes Logging (Duplicates)

**Problem:** Seeing duplicate log entries in distributed training.

**Solution:** Ensure `@on_main_process` decorator is used (happens automatically):
```python
# Accelerate handles this automatically
accelerator.log({"loss": loss}, step=step)  # Only logs on rank 0
```

---

### Issue 4: WandB Offline Mode Not Syncing

**Problem:** Trained in offline mode, want to sync to cloud.

**Solution:**
```bash
# Find offline run directory
ls wandb/

# Sync specific run
wandb sync wandb/offline-run-20231115_123456-abc123

# Sync all offline runs
wandb sync wandb/
```

---

### Issue 5: MLflow Experiment Not Found

**Problem:** `mlflow.search_experiments()` returns empty list.

**Solution:** Create experiment first:
```python
import mlflow

# Check existing experiments
mlflow.search_experiments()

# Create if needed
mlflow.create_experiment("my_experiment", artifact_location="./mlruns")
```

---

## Best Practices

### 1. Organize Metrics with Prefixes

```python
accelerator.log({
    # Training metrics
    "train/loss": train_loss,
    "train/accuracy": train_acc,

    # Validation metrics
    "val/loss": val_loss,
    "val/accuracy": val_acc,

    # System metrics
    "system/gpu_memory": gpu_memory,
    "system/learning_rate": lr,
}, step=step)
```

**Benefit:** Easy filtering in tracker UI (e.g., view all "train/*" metrics).

---

### 2. Log Hyperparameters at Start

```python
config = {
    "model": "bert-base-uncased",
    "learning_rate": 1e-4,
    "batch_size": 32,
    "num_epochs": 3,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "warmup_steps": 100,
}

accelerator.init_trackers("my_project", config=config)
```

**Benefit:** Easily compare experiments based on hyperparameters.

---

### 3. Use Consistent Step Granularity

```python
# Option A: Log every iteration (fine-grained)
for step, batch in enumerate(train_dataloader):
    loss = train_step(batch)
    accelerator.log({"train/loss": loss}, step=step)

# Option B: Log every epoch (coarse-grained)
for epoch in range(num_epochs):
    avg_loss = train_epoch(model, train_dataloader)
    accelerator.log({"train/loss": avg_loss}, step=epoch)
```

**Recommendation:** Iteration-level for training loss, epoch-level for validation.

---

### 4. Always Call end_training()

```python
try:
    # Training loop
    for epoch in range(num_epochs):
        train_epoch(model, dataloader)
finally:
    # Ensures logs are flushed even if training crashes
    accelerator.end_training()
```

---

### 5. Use Multiple Trackers for Redundancy

```python
accelerator = Accelerator(
    log_with=["tensorboard", "wandb"],  # Local + Cloud
)
```

**Benefit:** TensorBoard for local debugging, WandB for sharing/collaboration.

---

## Summary

### Key Methods

| Method | Purpose | Timing |
|--------|---------|--------|
| `Accelerator(log_with=...)` | Specify tracker(s) | Accelerator init |
| `init_trackers(project_name, config)` | Initialize tracking | After `prepare()`, before training |
| `log(values, step)` | Log metrics | During training |
| `get_tracker(name, unwrap)` | Access native tracker API | Anytime after `init_trackers()` |
| `end_training()` | Cleanup and finalize | End of script |

### Supported Trackers

| Tracker | Cloud/Local | Requires Dir | Special Features |
|---------|-------------|--------------|------------------|
| TensorBoard | Local | Yes | Graph visualization, embeddings |
| WandB | Cloud | No | Tables, artifacts, sweeps |
| Trackio | Cloud | No | Gradio integration |
| Comet ML | Cloud | No | Model registry |
| Aim | Local | Yes | Notebook integration |
| MLflow | Both | No | Model registry, deployment |
| ClearML | Cloud | No | Pipeline automation |
| DVCLive | Local | No | DVC integration |
| SwanLab | Cloud | No | Chinese UI |

### File References

- **Accelerator methods**: `src/accelerate/accelerator.py:3161-3292`
  - `init_trackers()`: Line 3161
  - `log()`: Line 3244
  - `get_tracker()`: Line 3211
  - `end_training()`: Line 3273
- **Tracker implementations**: `src/accelerate/tracking.py`

### Next Steps

- **14_Tracker_Feature_Comparison.md** - Detailed feature comparison matrix
- **15_Multi_Tracker_Best_Practices.md** - Advanced multi-tracker strategies
