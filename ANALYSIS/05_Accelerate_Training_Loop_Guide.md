# Data Flow & Lifecycle: Accelerate Training Loop Guide

## Overview
This guide traces the **complete data flow** of a training loop when using Accelerate. We'll follow a single batch from dataloader through forward pass, backward pass, optimizer step, and back to the next iteration.

---

## Standard Training Loop with Accelerate

```python
from accelerate import Accelerator

# 1. INITIALIZATION
accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=4
)

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
dataloader = DataLoader(dataset, batch_size=32)

# 2. PREPARATION (Injection Point)
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# 3. TRAINING LOOP
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        # A. FORWARD PASS
        outputs = model(batch)
        loss = loss_fn(outputs, labels)

        # B. BACKWARD PASS
        accelerator.backward(loss)

        # C. GRADIENT CLIPPING (optional)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

        # D. OPTIMIZER STEP
        optimizer.step()

        # E. ZERO GRADIENTS
        optimizer.zero_grad()

# 4. CLEANUP
accelerator.wait_for_everyone()
```

---

## Detailed Data Flow: Iteration-by-Iteration

### Scenario Setup
- **Hardware:** 4 GPUs (NVIDIA A100)
- **Distributed Type:** DDP (DistributedDataParallel)
- **Mixed Precision:** FP16
- **Gradient Accumulation:** 4 steps
- **Batch Size (per GPU):** 32
- **Total Batch Size:** 32 × 4 GPUs = 128
- **Effective Batch Size (with grad accumulation):** 128 × 4 = 512

---

## Phase 0: Initialization & Preparation

### Step 0.1: AcceleratorState Initialization
**Location:** `src/accelerate/state.py:PartialState.__init__`

```python
# When Accelerator() is created:
accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=4)

# Internally:
self.state = AcceleratorState(...)
    ↓
# AcceleratorState detects environment:
WORLD_SIZE = 4 (from environment variable)
RANK = 0, 1, 2, 3 (different on each GPU)
LOCAL_RANK = 0, 1, 2, 3

# Initialize process group:
torch.distributed.init_process_group(
    backend="nccl",
    init_method="env://",
    world_size=4,
    rank=self.process_index
)

# Result:
self.distributed_type = DistributedType.MULTI_GPU
self.device = torch.device("cuda", LOCAL_RANK)  # cuda:0, cuda:1, cuda:2, cuda:3
self.num_processes = 4
```

### Step 0.2: GradScaler Creation
**Location:** `src/accelerate/accelerator.py:562`

```python
# FP16 mixed precision is enabled, so create GradScaler:
if self.state.mixed_precision == "fp16":
    self.scaler = torch.cuda.amp.GradScaler()
    # Initial scale factor: 2^16 = 65536
```

### Step 0.3: Object Preparation
**Location:** `src/accelerate/accelerator.py:1413`

```python
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# What happens:
# 1. MODEL WRAPPING
model = prepare_model(model)
    → model.forward wrapped with autocast (FP16)
    → model.to(device)  # cuda:0, cuda:1, cuda:2, cuda:3
    → model = DistributedDataParallel(model, device_ids=[LOCAL_RANK], ...)
       # Registers backward hooks for gradient all-reduce

# 2. OPTIMIZER WRAPPING
optimizer = AcceleratedOptimizer(optimizer, scaler=self.scaler)
    → optimizer.state moved to device
    → optimizer.step patched to handle gradient accumulation + scaling

# 3. DATALOADER WRAPPING
dataloader = prepare_data_loader(dataloader)
    → Sampler replaced with DistributedSampler
       # Ensures each GPU gets different data
    → Wrapped in DataLoaderDispatcher (handles device placement)
```

**State After Preparation:**
```
GPU 0: model (full copy), optimizer, DistributedSampler (indices 0, 4, 8, ...)
GPU 1: model (full copy), optimizer, DistributedSampler (indices 1, 5, 9, ...)
GPU 2: model (full copy), optimizer, DistributedSampler (indices 2, 6, 10, ...)
GPU 3: model (full copy), optimizer, DistributedSampler (indices 3, 7, 11, ...)

All models have identical parameters (broadcasted from rank 0)
```

---

## Phase 1: Data Loading

### Step 1.1: Fetch Batch from DataLoader
**Location:** `src/accelerate/data_loader.py:DataLoaderDispatcher.__iter__`

```python
for step, batch in enumerate(dataloader):
```

**What Happens:**
```python
# On each GPU, DistributedSampler returns different indices:
# GPU 0: batch = dataset[0:32]    # Indices 0-31
# GPU 1: batch = dataset[32:64]   # Indices 32-63
# GPU 2: batch = dataset[64:96]   # Indices 64-95
# GPU 3: batch = dataset[96:128]  # Indices 96-127

# DataLoaderDispatcher moves batch to device:
batch = {k: v.to(device) for k, v in batch.items()}
# GPU 0: batch on cuda:0
# GPU 1: batch on cuda:1
# GPU 2: batch on cuda:2
# GPU 3: batch on cuda:3
```

**State:**
```
GPU 0: batch[0:32] on cuda:0
GPU 1: batch[32:64] on cuda:1
GPU 2: batch[64:96] on cuda:2
GPU 3: batch[96:128] on cuda:3

No inter-GPU communication yet
```

---

## Phase 2: Forward Pass (Iteration 1, Gradient Accumulation Step 1)

### Step 2.1: Model Forward
**Location:** `src/accelerate/accelerator.py:1750` (model.forward wrapping)

```python
outputs = model(batch)
```

**What Happens:**
```python
# model.forward is wrapped with autocast:
with torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = model._original_forward(batch)

# On each GPU independently:
# GPU 0: outputs_0 = model(batch[0:32])    # FP16 activations
# GPU 1: outputs_1 = model(batch[32:64])   # FP16 activations
# GPU 2: outputs_2 = model(batch[64:96])   # FP16 activations
# GPU 3: outputs_3 = model(batch[96:128])  # FP16 activations

# Each GPU computes forward pass INDEPENDENTLY
# No inter-GPU communication during forward pass
```

**Autocast Effect:**
```python
# Linear layers: FP32 weights × FP16 inputs → FP16 outputs
# Activations: Computed in FP16
# Layer norm, batch norm: FP32 (for numerical stability)
# Loss functions: FP32
```

### Step 2.2: Loss Computation
```python
loss = loss_fn(outputs, labels)
```

**What Happens:**
```python
# On each GPU:
# GPU 0: loss_0 = CrossEntropyLoss(outputs_0, labels[0:32])   # FP32 scalar
# GPU 1: loss_1 = CrossEntropyLoss(outputs_1, labels[32:64])  # FP32 scalar
# GPU 2: loss_2 = CrossEntropyLoss(outputs_2, labels[64:96])  # FP32 scalar
# GPU 3: loss_3 = CrossEntropyLoss(outputs_3, labels[96:128]) # FP32 scalar

# Each GPU has a DIFFERENT loss value (different data)
# Losses are in FP32 (autocast converts outputs to FP32 before loss)
```

**State:**
```
GPU 0: loss_0 (e.g., 2.345), outputs_0 in FP16
GPU 1: loss_1 (e.g., 2.389), outputs_1 in FP16
GPU 2: loss_2 (e.g., 2.301), outputs_2 in FP16
GPU 3: loss_3 (e.g., 2.412), outputs_3 in FP16
```

---

## Phase 3: Backward Pass (Gradient Computation + Synchronization)

### Step 3.1: Accelerator.backward()
**Location:** `src/accelerate/accelerator.py:2708`

```python
accelerator.backward(loss)
```

**What Happens:**
```python
# Step 1: Gradient Accumulation Scaling
loss = loss / self.gradient_accumulation_steps  # loss / 4

# Step 2: Loss Scaling (FP16)
scaled_loss = self.scaler.scale(loss)
# scaled_loss = loss * 65536 (initial scale factor)

# Step 3: Backward Pass
scaled_loss.backward()
```

### Step 3.2: Gradient Computation (Per-GPU)
```python
# On each GPU:
# GPU 0: loss_0 / 4 → scaled_loss_0 → backward()
#        Computes gradients in FP16 (scaled by 65536 / 4)
# GPU 1: loss_1 / 4 → scaled_loss_1 → backward()
# GPU 2: loss_2 / 4 → scaled_loss_2 → backward()
# GPU 3: loss_3 / 4 → scaled_loss_3 → backward()

# Each GPU computes gradients for its batch independently
# Gradients stored in param.grad (FP16, scaled)
```

### Step 3.3: DDP Gradient Synchronization (Automatic)
**Location:** PyTorch DDP hooks (registered during prepare_model)

```python
# DDP backward hooks trigger automatically during backward():
for param in model.parameters():
    if param.requires_grad:
        # Hook fires after param.grad is computed
        # 1. All-reduce gradient across all GPUs
        torch.distributed.all_reduce(param.grad, op=ReduceOp.SUM)

        # 2. Average gradient
        param.grad /= world_size  # param.grad /= 4

# Result: All GPUs have IDENTICAL averaged gradients
```

**Gradient Synchronization Example:**
```
Before all-reduce:
  GPU 0: param.grad = [1.0, 2.0, 3.0]  (scaled by 65536/4)
  GPU 1: param.grad = [1.1, 2.1, 3.1]
  GPU 2: param.grad = [0.9, 1.9, 2.9]
  GPU 3: param.grad = [1.0, 2.0, 3.0]

After all-reduce (SUM):
  All GPUs: param.grad = [4.0, 8.0, 12.0]

After averaging (/4):
  All GPUs: param.grad = [1.0, 2.0, 3.0]  (averaged, still scaled by 65536/4)
```

**State After Backward:**
```
All GPUs: Identical gradients in param.grad (FP16, scaled by 65536/4)
gradient_state.sync_gradients = False (accumulation step 1 of 4)
```

---

## Phase 4: Gradient Accumulation (Steps 2, 3, 4)

### Iteration 2, 3, 4: Repeat Phase 1-3
```python
# Step 2 of gradient accumulation:
batch = next(dataloader)  # New batch for each GPU
outputs = model(batch)
loss = loss_fn(outputs, labels)
accelerator.backward(loss)  # Gradients ADDED to existing param.grad

# Step 3 of gradient accumulation:
# ... same as above ...

# Step 4 of gradient accumulation:
# ... same as above ...

# After 4 accumulation steps:
# param.grad contains sum of 4 mini-batch gradients (still scaled by 65536/4)
```

**Gradient Accumulation Logic:**
```python
# Accelerate automatically handles accumulation via sync_gradients flag:
if step % gradient_accumulation_steps == 0:
    gradient_state.sync_gradients = True  # Enable optimizer step
else:
    gradient_state.sync_gradients = False  # Skip optimizer step
```

**State After 4 Accumulation Steps:**
```
All GPUs: param.grad = sum of 4 mini-batch gradients (scaled by 65536/4)
gradient_state.sync_gradients = True
accelerator.sync_gradients = True
```

---

## Phase 5: Gradient Clipping (Optional)

### Step 5.1: Accelerator.clip_grad_norm_()
**Location:** `src/accelerate/accelerator.py:2836`

```python
if accelerator.sync_gradients:  # True on accumulation step 4
    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**What Happens:**
```python
# Step 1: Unscale Gradients
self.scaler.unscale_(optimizer)
# Divides all param.grad by scale_factor (65536)
# Now gradients are true FP32 gradients (no longer scaled)

# Step 2: Compute Global Gradient Norm
total_norm = 0.0
for param in model.parameters():
    if param.grad is not None:
        param_norm = param.grad.data.norm(2)  # L2 norm
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5  # Square root

# Step 3: Clip if Necessary
if total_norm > max_norm:
    clip_coef = max_norm / (total_norm + 1e-6)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.mul_(clip_coef)

# Result: Gradients are clipped to max_norm=1.0
```

**State After Clipping:**
```
All GPUs: param.grad = clipped, averaged, unscaled gradients (FP32)
```

---

## Phase 6: Optimizer Step

### Step 6.1: AcceleratedOptimizer.step()
**Location:** `src/accelerate/optimizer.py:145`

```python
optimizer.step()
```

**What Happens:**
```python
# Check gradient accumulation flag:
if self.gradient_state.sync_gradients:  # True
    if self.scaler is not None:  # True (FP16 mode)
        # ScalerStep:
        # 1. Gradients already unscaled (during clipping)
        # 2. Check for inf/NaN in gradients
        found_inf = check_for_inf_or_nan(gradients)

        if not found_inf:
            # 3. Step optimizer (update parameters)
            self.optimizer.step()
            # Each GPU updates its parameters independently
            # Since gradients are identical, parameters stay synchronized

            # 4. Update scale factor (increase if no overflow)
            self.scaler.update()
            # scale_factor *= 2 (up to max)
        else:
            # Overflow detected: skip step, reduce scale
            self.scaler.update()
            # scale_factor /= 2
            # Parameters NOT updated
```

**Parameter Update (AdamW Example):**
```python
# For each parameter on each GPU:
# 1. Compute moments (m_t, v_t)
# 2. Update parameter:
#    param = param - lr * m_t / (sqrt(v_t) + eps)

# All GPUs have identical gradients, so all updates are identical
# Parameters remain synchronized across GPUs
```

**State After Step:**
```
All GPUs: Parameters updated (FP32), synchronized
gradient_state.sync_gradients = False (reset for next accumulation cycle)
```

---

## Phase 7: Zero Gradients

### Step 7.1: AcceleratedOptimizer.zero_grad()
**Location:** `src/accelerate/optimizer.py:112`

```python
optimizer.zero_grad()
```

**What Happens:**
```python
# Only zeros gradients if sync_gradients was True
if self.gradient_state.sync_gradients:  # Now False
    # Skip zeroing (we're in accumulation mode for next iteration)
    pass

# Gradients are PRESERVED for next accumulation step
```

**After 4th Accumulation Step:**
```python
# sync_gradients becomes True again, so:
if self.gradient_state.sync_gradients:  # True
    self.optimizer.zero_grad(set_to_none=True)
    # All param.grad set to None
```

**State After Zero Grad:**
```
All GPUs: param.grad = None (ready for next batch)
```

---

## Phase 8: Next Iteration (Cycle Repeats)

### Gradient Accumulation Cycle
```
Iteration 1:  Forward → Backward → Accumulate (step 1/4) → Skip optimizer → Keep gradients
Iteration 2:  Forward → Backward → Accumulate (step 2/4) → Skip optimizer → Keep gradients
Iteration 3:  Forward → Backward → Accumulate (step 3/4) → Skip optimizer → Keep gradients
Iteration 4:  Forward → Backward → Accumulate (step 4/4) → Clip → Optimizer step → Zero grad

[Cycle repeats]
```

---

## Communication Timeline: DDP with FP16 + Gradient Accumulation

### Iteration 1 (Accumulation Step 1/4)

| **Time** | **GPU 0** | **GPU 1** | **GPU 2** | **GPU 3** | **Communication** |
|----------|-----------|-----------|-----------|-----------|-------------------|
| T0 | Load batch[0:32] | Load batch[32:64] | Load batch[64:96] | Load batch[96:128] | None |
| T1 | Forward (FP16) | Forward (FP16) | Forward (FP16) | Forward (FP16) | None |
| T2 | Loss (FP32) | Loss (FP32) | Loss (FP32) | Loss (FP32) | None |
| T3 | Backward | Backward | Backward | Backward | **All-reduce gradients** |
| T4 | grad accumulated | grad accumulated | grad accumulated | grad accumulated | None |

### Iteration 2-3 (Accumulation Steps 2-3/4)
Same as Iteration 1

### Iteration 4 (Accumulation Step 4/4)

| **Time** | **GPU 0** | **GPU 1** | **GPU 2** | **GPU 3** | **Communication** |
|----------|-----------|-----------|-----------|-----------|-------------------|
| T0 | Load batch | Load batch | Load batch | Load batch | None |
| T1 | Forward | Forward | Forward | Forward | None |
| T2 | Loss | Loss | Loss | Loss | None |
| T3 | Backward | Backward | Backward | Backward | **All-reduce gradients** |
| T4 | Unscale | Unscale | Unscale | Unscale | None |
| T5 | Clip grads | Clip grads | Clip grads | Clip grads | None |
| T6 | Optimizer step | Optimizer step | Optimizer step | Optimizer step | None |
| T7 | Zero grad | Zero grad | Zero grad | Zero grad | None |

**Communication Overhead:**
- **All-reduce:** Happens 4 times per effective batch (once per accumulation step)
- **Bandwidth Used:** ~2 × model_size per all-reduce (send + receive)
- **Overlapping:** DDP overlaps all-reduce with backward pass (reduces latency)

---

## Key Insights

### 1. Transparent Interception Points
User calls standard PyTorch methods, but Accelerate intercepts:
- `model(batch)` → Autocast context + DDP forward hooks
- `loss.backward()` → DDP gradient synchronization hooks
- `optimizer.step()` → Gradient accumulation gating + scaler step
- `optimizer.zero_grad()` → Gradient accumulation gating

### 2. Gradient Synchronization Guarantee
```python
# DDP guarantees: After loss.backward() completes
# → All GPUs have IDENTICAL gradients

# Proof: Assume GPUs have identical parameters before step
# → Same parameters + identical gradients → identical updates
# → Parameters remain identical after step
# → Invariant maintained throughout training
```

### 3. Memory vs. Computation Trade-off

| **Config** | **Memory per GPU** | **Computation** | **Communication** |
|------------|-------------------|----------------|-------------------|
| No gradient accumulation | High (large batch) | Fast (fewer steps) | Less (fewer all-reduces) |
| 4x gradient accumulation | Low (small batch) | Slower (more steps) | More (more all-reduces) |

### 4. Critical Code Paths

| **Operation** | **File** | **Line** |
|--------------|----------|----------|
| Dataloader iteration | `src/accelerate/data_loader.py` | DataLoaderDispatcher.__iter__ |
| Autocast wrapping | `src/accelerate/accelerator.py` | 1750-1761 |
| Backward pass | `src/accelerate/accelerator.py` | 2708-2740 |
| DDP gradient sync | PyTorch DDP | Backward hooks |
| Gradient clipping | `src/accelerate/accelerator.py` | 2836-2897 |
| Optimizer step | `src/accelerate/optimizer.py` | 145-200 |
| Gradient accumulation | `src/accelerate/state.py` | GradientState |

---

## Comparison: Standard PyTorch vs. Accelerate

### Standard PyTorch DDP (Manual Setup)
```python
# User must handle:
# 1. torch.distributed.init_process_group()
# 2. model = DistributedDataParallel(model, device_ids=[rank])
# 3. sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
# 4. Manual gradient accumulation logic
# 5. Manual mixed precision (autocast, scaler.scale, scaler.step, scaler.update)
# 6. Manual gradient clipping with scaler.unscale_()
# 7. Manual device placement
```

### Accelerate (Automatic Setup)
```python
# Accelerate handles automatically:
# 1. Environment detection + init_process_group
# 2. Model wrapping (DDP/FSDP/DeepSpeed)
# 3. Distributed sampler injection
# 4. Gradient accumulation (via sync_gradients flag)
# 5. Mixed precision (autocast + scaler)
# 6. Gradient clipping (with automatic unscaling)
# 7. Device placement

# User code remains standard PyTorch!
```

---

## Summary: The Complete Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│ INITIALIZATION                                               │
│ 1. AcceleratorState detects environment (4 GPUs, DDP)       │
│ 2. GradScaler created for FP16                              │
│ 3. prepare() wraps model, optimizer, dataloader             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ ACCUMULATION STEP 1-3 (repeat 3 times)                      │
│ 1. Load batch (different data on each GPU)                  │
│ 2. Forward pass (FP16, independent per GPU)                 │
│ 3. Loss computation (FP32, different per GPU)               │
│ 4. Backward pass (FP16 gradients, all-reduced & averaged)   │
│ 5. Skip optimizer step (sync_gradients=False)               │
│ 6. Keep gradients for accumulation                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ ACCUMULATION STEP 4 (optimizer step)                        │
│ 1. Load batch                                                │
│ 2. Forward pass (FP16)                                       │
│ 3. Loss computation (FP32)                                   │
│ 4. Backward pass (gradients all-reduced)                    │
│ 5. Unscale gradients (FP32)                                  │
│ 6. Clip gradients (max_norm=1.0)                            │
│ 7. Optimizer step (update parameters)                       │
│ 8. Zero gradients                                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
                   [Cycle repeats]
```

All of this complexity is **hidden** from the user - they just call standard PyTorch methods!
