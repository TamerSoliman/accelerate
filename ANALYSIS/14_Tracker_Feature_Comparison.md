# Integration 3: Tracker Feature Comparison

## Overview

This document provides a comprehensive comparison of the 9 experiment tracking platforms supported by Accelerate, helping you choose the right tracker(s) for your use case.

---

## Quick Comparison Matrix

### Core Features

| Feature | TensorBoard | WandB | Trackio | Comet ML | Aim | MLflow | ClearML | DVCLive | SwanLab |
|---------|------------|-------|---------|----------|-----|--------|---------|---------|---------|
| **Scalar Metrics** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Text Logging** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| **Dict Metrics** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| **Images** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Tables** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Hyperparameters** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Model Graphs** | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Artifacts** | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |

### Infrastructure

| Feature | TensorBoard | WandB | Trackio | Comet ML | Aim | MLflow | ClearML | DVCLive | SwanLab |
|---------|------------|-------|---------|----------|-----|--------|---------|---------|---------|
| **Requires Directory** | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Cloud-Based** | ❌ | ✅ | ✅ | ✅ | ❌ | Optional | ✅ | ❌ | ✅ |
| **Self-Hosted** | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | Optional | ✅ | ❌ |
| **Offline Mode** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Main Process Only** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

### Advanced Features

| Feature | TensorBoard | WandB | Trackio | Comet ML | Aim | MLflow | ClearML | DVCLive | SwanLab |
|---------|------------|-------|---------|----------|-----|--------|---------|---------|---------|
| **Model Registry** | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Hyperparameter Sweeps** | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Collaboration** | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ |
| **API Access** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Embeddings Visualization** | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

---

## Detailed Feature Analysis

### 1. TensorBoard

**Best For:** Local debugging, PyTorch users, integration with existing workflows

**Strengths:**
- ✅ **Native PyTorch integration** - Ships with `torch.utils.tensorboard`
- ✅ **Rich visualizations** - Scalars, images, histograms, graphs, embeddings (t-SNE/PCA)
- ✅ **Offline by default** - No cloud dependency
- ✅ **Free and open-source** - No limits
- ✅ **Model graph visualization** - `add_graph()` shows architecture

**Limitations:**
- ❌ **No cloud sync** - Requires manual file sharing
- ❌ **No model registry** - Cannot version models
- ❌ **No collaboration tools** - No comments, sharing, or team features
- ❌ **File-based storage** - Can become slow with many experiments

**Data Types Supported:**
```python
# Scalars
writer.add_scalar("loss", loss, step)

# Multiple scalars (grouped)
writer.add_scalars("losses", {"train": train_loss, "val": val_loss}, step)

# Text
writer.add_text("predictions", "Sample: positive", step)

# Images (as tensors)
writer.add_image("sample", img_tensor, step)
writer.add_images("batch", img_grid, step)

# Histograms
writer.add_histogram("weights", model.layer.weight, step)

# Model graph
writer.add_graph(model, input_sample)

# Embeddings (t-SNE)
writer.add_embedding(embeddings, metadata=labels, label_img=images)

# Hyperparameters
writer.add_hparams(hparams, {"val/loss": val_loss})
```

**Accelerate Integration:**
```python
accelerator = Accelerator(
    log_with="tensorboard",
    project_dir="./logs",  # Required
)
```

**Viewing Logs:**
```bash
tensorboard --logdir=./logs/my_project
```

---

### 2. Weights & Biases (WandB)

**Best For:** Team collaboration, experiment tracking at scale, model versioning

**Strengths:**
- ✅ **Cloud-based** - Access experiments from anywhere
- ✅ **Rich visualizations** - Interactive plots, media logging (images, audio, video, 3D)
- ✅ **Collaboration features** - Comments, sharing, reports
- ✅ **Hyperparameter sweeps** - Built-in bayesian optimization
- ✅ **Model registry** - Version and deploy models
- ✅ **Artifacts** - Track datasets, models, intermediate outputs
- ✅ **Tables** - Log structured data with rich types

**Limitations:**
- ❌ **Cloud dependency** - Offline mode available but limited
- ❌ **Pricing** - Free tier has limits (100GB storage, 7-day retention)
- ❌ **Privacy concerns** - Data stored on WandB servers

**Data Types Supported:**
```python
import wandb

# Scalars
wandb.log({"loss": loss}, step=step)

# Images
wandb.log({"examples": [wandb.Image(img) for img in images]}, step=step)

# Tables
table = wandb.Table(columns=["image", "pred", "label"], data=data)
wandb.log({"predictions": table}, step=step)

# Audio
wandb.log({"speech": wandb.Audio(audio_array, sample_rate=16000)}, step=step)

# Videos
wandb.log({"video": wandb.Video(video_array, fps=30)}, step=step)

# 3D Point Clouds
wandb.log({"scene": wandb.Object3D(point_cloud)}, step=step)

# HTML
wandb.log({"custom_plot": wandb.Html("<div>...</div>")}, step=step)

# Molecules
wandb.log({"molecule": wandb.Molecule("molecule.pdb")}, step=step)
```

**Accelerate Integration:**
```python
accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(
    "my_project",
    init_kwargs={
        "wandb": {
            "entity": "my_team",
            "tags": ["bert", "v1"],
            "notes": "Baseline experiment",
        }
    },
)
```

**Special Features:**
```python
# Watch model (log gradients and parameters)
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
wandb_tracker.watch(model, log="all", log_freq=100)

# Save artifacts
artifact = wandb.Artifact("dataset", type="dataset")
artifact.add_dir("data/")
wandb_tracker.log_artifact(artifact)

# Alerts
wandb_tracker.alert(title="Training Complete", text="Loss: 0.001")
```

---

### 3. MLflow

**Best For:** Production ML workflows, model deployment, end-to-end ML lifecycle

**Strengths:**
- ✅ **Model registry** - Version and stage models (Staging → Production)
- ✅ **Model deployment** - Serve models via REST API
- ✅ **Flexible backend** - Local files, database, cloud (S3, Azure, GCS)
- ✅ **Framework agnostic** - Works with PyTorch, TensorFlow, Scikit-learn, etc.
- ✅ **Artifact tracking** - Log models, plots, data files
- ✅ **Environment reproducibility** - Captures dependencies

**Limitations:**
- ❌ **Only numeric metrics** - No text or dict logging in `log()`
- ❌ **Parameter length limits** - Max 500 characters per parameter
- ❌ **Batch size limits** - Max 100 parameters/tags per batch
- ❌ **Basic UI** - Less interactive than WandB

**Data Types Supported:**
```python
import mlflow

# Scalars (int/float only)
mlflow.log_metric("loss", loss, step=step)
mlflow.log_metrics({"loss": loss, "accuracy": acc}, step=step)

# Parameters (strings up to 500 chars)
mlflow.log_param("learning_rate", 1e-4)
mlflow.log_params({"lr": 1e-4, "batch_size": 32})

# Tags
mlflow.set_tag("model_type", "bert")
mlflow.set_tags({"version": "v1.0", "env": "production"})

# Artifacts (files/directories)
mlflow.log_artifact("model.pt")  # Single file
mlflow.log_artifacts("outputs/")  # Entire directory

# Models (with signature)
mlflow.pytorch.log_model(model, "model", registered_model_name="bert-classifier")

# Figures (matplotlib, plotly)
mlflow.log_figure(plt.gcf(), "loss_curve.png")
```

**Accelerate Integration:**
```python
accelerator = Accelerator(log_with="mlflow")
accelerator.init_trackers(
    "my_experiment",
    init_kwargs={
        "mlflow": {
            "run_name": "run_001",
            "tags": {"version": "v1", "dataset": "imdb"},
            "description": "Baseline BERT",
        }
    },
)
```

**Special Features:**
```python
# Search experiments
experiments = mlflow.search_experiments(filter_string="name = 'my_exp'")

# Search runs
runs = mlflow.search_runs(experiment_ids=["1"], filter_string="metrics.loss < 0.5")

# Load model from registry
model = mlflow.pytorch.load_model("models:/bert-classifier/Production")

# Serve model
# mlflow models serve -m models:/bert-classifier/Production -p 5000
```

---

### 4. Comet ML

**Best For:** Enterprise teams, advanced experiment management, model explainability

**Strengths:**
- ✅ **Comprehensive logging** - Metrics, parameters, system metrics, code, dependencies
- ✅ **Model registry** - Version and manage models
- ✅ **Experiment comparison** - Side-by-side comparison of runs
- ✅ **Code tracking** - Automatically logs git hash, diffs
- ✅ **System monitoring** - GPU/CPU usage, memory
- ✅ **Optimization** - Hyperparameter search

**Limitations:**
- ❌ **Pricing** - Free tier limited (1000 experiments/month)
- ❌ **Cloud dependency** - Self-hosted option expensive

**Data Types Supported:**
```python
from comet_ml import Experiment

# Scalars
experiment.log_metric("loss", loss, step=step)

# Text
experiment.log_other("prediction", "positive")

# Parameters
experiment.log_parameters({"lr": 1e-4, "batch_size": 32})

# Images
experiment.log_image("sample.png", image_data)

# Models
experiment.log_model("bert-model", "model.pt")

# Code
experiment.log_code()  # Logs git hash and diff

# System metrics (automatic)
# GPU usage, CPU usage, RAM, etc.
```

**Accelerate Integration:**
```python
accelerator = Accelerator(log_with="comet_ml")
accelerator.init_trackers("my_project")
```

---

### 5. Aim

**Best For:** Local experiment tracking with powerful querying, research teams

**Strengths:**
- ✅ **Fast local storage** - Optimized for large-scale experiment tracking
- ✅ **Powerful query language** - SQL-like syntax for filtering runs
- ✅ **Interactive UI** - Jupyter notebook integration
- ✅ **Open source** - No cloud dependency

**Limitations:**
- ❌ **No cloud sync** - Manual sharing required
- ❌ **Limited collaboration** - No built-in team features

**Data Types Supported:**
```python
from aim import Run

run = Run()

# Scalars
run.track(loss, name="loss", step=step)

# Context (grouping)
run.track(loss, name="loss", step=step, context={"subset": "train"})

# Images
from aim import Image
run.track(Image(img), name="sample", step=step)

# Distributions
from aim import Distribution
run.track(Distribution(values), name="weights", step=step)

# Hyperparameters
run["hparams"] = {"lr": 1e-4, "batch_size": 32}
```

**Accelerate Integration:**
```python
accelerator = Accelerator(
    log_with="aim",
    project_dir="./aim_logs",  # Required
)
```

**Viewing Logs:**
```bash
aim up  # Opens UI at http://localhost:43800
```

---

### 6. ClearML

**Best For:** MLOps pipelines, automated workflows, enterprise deployments

**Strengths:**
- ✅ **Full MLOps platform** - Experiment tracking + pipeline orchestration + model serving
- ✅ **Auto-logging** - Captures code, environment, git info automatically
- ✅ **Pipeline automation** - Create DAGs of training tasks
- ✅ **Resource management** - Schedule jobs on GPU clusters

**Limitations:**
- ❌ **Complexity** - Steeper learning curve
- ❌ **Pricing** - Free tier limited

**Data Types Supported:**
```python
from clearml import Task

task = Task.init(project_name="my_project", task_name="train")

# Scalars (auto-parses train/val/test prefixes)
logger = task.get_logger()
logger.report_scalar("loss", "train", value=loss, iteration=step)

# Images
logger.report_image("sample", "train", image=img, iteration=step)

# Tables
logger.report_table("predictions", "data", table_plot=dataframe, iteration=step)

# Configuration (auto-captures argparse)
task.connect_configuration(config_dict)
```

**Accelerate Integration:**
```python
accelerator = Accelerator(log_with="clearml")
accelerator.init_trackers("my_project")
```

---

### 7. DVCLive

**Best For:** Integration with DVC (Data Version Control), reproducible experiments

**Strengths:**
- ✅ **DVC integration** - Automatically logs to DVC experiments
- ✅ **Git-based** - Experiments tracked in git
- ✅ **Lightweight** - Minimal overhead
- ✅ **Open source**

**Limitations:**
- ❌ **Only scalar metrics** - No images, text, or complex types
- ❌ **Limited UI** - Basic visualization

**Data Types Supported:**
```python
from dvclive import Live

live = Live()

# Scalars (only numeric types)
live.log_metric("loss", loss)

# Parameters
live.log_params({"lr": 1e-4, "batch_size": 32})

# Next step
live.next_step()
```

**Accelerate Integration:**
```python
accelerator = Accelerator(log_with="dvclive")
accelerator.init_trackers("my_project")
```

---

### 8. SwanLab

**Best For:** Chinese ML teams, WandB alternative with Chinese UI

**Strengths:**
- ✅ **Chinese UI** - Native Chinese language support
- ✅ **WandB-like features** - Similar API and capabilities
- ✅ **Cloud-based** - Collaborative tracking

**Limitations:**
- ❌ **Newer platform** - Less mature than WandB/MLflow
- ❌ **Documentation** - Primarily in Chinese

**Data Types Supported:**
```python
import swanlab

swanlab.init(project="my_project")

# Scalars
swanlab.log({"loss": loss}, step=step)

# Images
swanlab.log({"samples": [swanlab.Image(img) for img in images]}, step=step)

# Configuration
swanlab.config.update({"lr": 1e-4})
```

**Accelerate Integration:**
```python
accelerator = Accelerator(log_with="swanlab")
accelerator.init_trackers("my_project")
```

---

### 9. Trackio

**Best For:** Gradio users, simple experiment tracking with Gradio integration

**Strengths:**
- ✅ **Gradio integration** - Built by Gradio team
- ✅ **Simple API** - Similar to WandB
- ✅ **Cloud-based**

**Limitations:**
- ❌ **Limited features** - Fewer capabilities than WandB
- ❌ **Newer platform** - Less mature

**Data Types Supported:**
```python
import trackio

trackio.init(project="my_project")

# Scalars
trackio.log({"loss": loss}, step=step)

# Configuration
trackio.config.update({"lr": 1e-4})
```

**Accelerate Integration:**
```python
accelerator = Accelerator(log_with="trackio")
accelerator.init_trackers("my_project")
```

---

## Use Case Recommendations

### Local Development & Debugging
**Recommended:** TensorBoard, Aim
- **Why:** Fast, offline, no cloud dependency
- **Setup:** `Accelerator(log_with="tensorboard", project_dir="./logs")`

### Team Collaboration
**Recommended:** WandB, ClearML, Comet ML
- **Why:** Cloud-based, sharing, commenting, reports
- **Setup:** `Accelerator(log_with="wandb")`

### Production ML Lifecycle
**Recommended:** MLflow, ClearML
- **Why:** Model registry, deployment, versioning
- **Setup:** `Accelerator(log_with="mlflow")`

### Hyperparameter Optimization
**Recommended:** WandB (Sweeps), Comet ML, ClearML
- **Why:** Built-in bayesian optimization
- **Setup:** Use WandB `wandb.sweep()` API

### Open Source / Self-Hosted
**Recommended:** TensorBoard, Aim, MLflow, DVCLive
- **Why:** No cloud dependency, full control
- **Setup:** All run locally

### Data Versioning Integration
**Recommended:** DVCLive, ClearML
- **Why:** Integrate with DVC/ClearML data management
- **Setup:** `Accelerator(log_with="dvclive")`

### Gradio Integration
**Recommended:** Trackio
- **Why:** Built by Gradio team, seamless integration
- **Setup:** `Accelerator(log_with="trackio")`

### Chinese Teams
**Recommended:** SwanLab
- **Why:** Native Chinese UI and documentation
- **Setup:** `Accelerator(log_with="swanlab")`

---

## Performance Comparison

### Logging Speed (Approximate)

| Tracker | Overhead | Notes |
|---------|----------|-------|
| TensorBoard | Low | File I/O can slow down with many metrics |
| WandB | Medium | Network latency for cloud uploads |
| Trackio | Medium | Cloud-based |
| Comet ML | Medium | Cloud-based |
| Aim | Low | Optimized local storage |
| MLflow | Low-Medium | Depends on backend (local vs server) |
| ClearML | Medium | Cloud-based |
| DVCLive | Low | Minimal overhead |
| SwanLab | Medium | Cloud-based |

**Tip:** For minimal overhead, use local trackers (TensorBoard, Aim, DVCLive)

---

### Storage Requirements

| Tracker | Storage Location | Typical Size (10k steps) |
|---------|-----------------|-------------------------|
| TensorBoard | `./logs/` | ~50 MB (scalars only) |
| WandB | WandB cloud | ~20 MB (cloud storage) |
| Aim | `./.aim/` | ~30 MB (optimized storage) |
| MLflow | `./mlruns/` or DB | ~40 MB |
| DVCLive | `./dvclive/` | ~10 MB (scalars only) |

---

## Cost Comparison

| Tracker | Free Tier | Paid Plans | Self-Hosted |
|---------|-----------|------------|-------------|
| TensorBoard | Unlimited | N/A | ✅ |
| WandB | 100 GB, 7-day retention | $50-$500/month | ❌ |
| Trackio | Unknown | Unknown | ❌ |
| Comet ML | 1000 exp/month | $29-$399/month | Enterprise only |
| Aim | Unlimited | N/A | ✅ |
| MLflow | Unlimited | N/A (managed by cloud provider) | ✅ |
| ClearML | 100 GB | $99-$999/month | ✅ (open source) |
| DVCLive | Unlimited | N/A | ✅ |
| SwanLab | Unknown | Unknown | ❌ |

---

## Migration Guide

### From TensorBoard to WandB

```python
# Before
accelerator = Accelerator(log_with="tensorboard", project_dir="./logs")
accelerator.init_trackers("my_project")

# After
accelerator = Accelerator(log_with="wandb")  # Remove project_dir
accelerator.init_trackers("my_project", init_kwargs={
    "wandb": {"entity": "my_team"}
})

# Logging code stays the same!
accelerator.log({"loss": loss}, step=step)
```

### From WandB to MLflow

```python
# Before
accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers("my_project")

# After
accelerator = Accelerator(log_with="mlflow")
accelerator.init_trackers("my_experiment")  # MLflow uses "experiment" terminology

# Logging code stays the same!
accelerator.log({"loss": loss}, step=step)
```

---

## Summary Table

| Tracker | Type | Best For | Requires Dir | Special Feature |
|---------|------|----------|--------------|-----------------|
| TensorBoard | Local | Debugging | ✅ | Model graphs, embeddings |
| WandB | Cloud | Collaboration | ❌ | Tables, sweeps, artifacts |
| Trackio | Cloud | Gradio users | ❌ | Gradio integration |
| Comet ML | Cloud | Enterprise | ❌ | Code tracking, model explainability |
| Aim | Local | Research | ✅ | Query language, fast storage |
| MLflow | Both | Production | ❌ | Model registry, deployment |
| ClearML | Cloud | MLOps | ❌ | Pipeline automation |
| DVCLive | Local | DVC users | ❌ | Git-based experiments |
| SwanLab | Cloud | Chinese teams | ❌ | Chinese UI |

---

## Decision Tree

```
Do you need cloud collaboration?
├─ Yes
│  ├─ Need model registry & deployment?
│  │  ├─ Yes → MLflow or ClearML
│  │  └─ No  → WandB or Comet ML
│  └─ Chinese team? → SwanLab
│
└─ No (local only)
   ├─ Need rich visualizations?
   │  ├─ Yes → TensorBoard
   │  └─ No  → DVCLive
   └─ Need fast querying? → Aim
```

---

## File References

- **Tracker implementations**: `src/accelerate/tracking.py:101-1268`
- **TensorBoard**: Lines 182-295
- **WandB**: Lines 297-429
- **MLflow**: Lines 705-910
- **Feature detection**: Lines 47-72 (availability checks)

---

## Next Steps

- **15_Multi_Tracker_Best_Practices.md** - Advanced strategies for using multiple trackers simultaneously
