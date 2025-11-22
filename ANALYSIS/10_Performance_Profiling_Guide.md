# Enhancement 4: Performance Profiling & Bottleneck Analysis

## Overview
Accelerate provides built-in profiling support through PyTorch's profiler, enabling identification of performance bottlenecks in distributed training.

---

## Using accelerator.profile()

### Basic Usage

```python
from accelerate import Accelerator
from accelerate.utils import ProfileKwargs

# Configure profiler
profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],          # Profile CPU and CUDA operations
    with_stack=True,                     # Include stack traces
    with_flops=True,                     # Compute FLOPs
    with_modules=True,                   # Include module names
    output_trace_dir="./profile_traces"  # Save Chrome traces here
)

accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

# Profile training loop
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

with accelerator.profile() as prof:
    for batch in dataloader:
        outputs = model(batch)
        loss = loss_fn(outputs, labels)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        prof.step()  # Mark step boundary

# Results saved to ./profile_traces/
```

---

## ProfileKwargs Configuration

**Location:** `src/accelerate/utils/dataclasses.py`

```python
@dataclass
class ProfileKwargs(KwargsHandler):
    activities: list[str] = field(default_factory=lambda: ["cpu", "cuda"])
    schedule_option: dict | None = None  # wait/warmup/active/repeat cycles
    on_trace_ready: Callable | None = None  # Custom trace handler
    record_shapes: bool = False  # Record tensor shapes
    profile_memory: bool = False  # Track memory allocations
    with_stack: bool = False  # Include Python stack traces
    with_flops: bool = False  # Compute FLOPs per operation
    with_modules: bool = False  # Include module hierarchy
    output_trace_dir: str | None = None  # Chrome trace output directory
```

---

## Advanced Profiling: Scheduling

```python
# Profile specific iterations only
profile_kwargs = ProfileKwargs(
    schedule_option=dict(
        wait=1,    # Skip first iteration
        warmup=1,  # Warmup iteration (not recorded)
        active=3,  # Profile next 3 iterations
        repeat=2   # Repeat cycle 2 times
    ),
    output_trace_dir="./traces"
)

# Total iterations: wait(1) + (warmup(1) + active(3)) × repeat(2) = 9 iterations
```

---

## Common Bottlenecks & Solutions

### 1. Data Loading Bottleneck

**Symptom:** CPU utilization low, GPU idle between batches

**Diagnosis:**
```python
# Look for large gaps between iterations in trace
with accelerator.profile() as prof:
    for batch in dataloader:
        # Profile shows: GPU idle 80% of time
        outputs = model(batch)
```

**Solutions:**
```python
# A. Increase num_workers
dataloader = DataLoader(dataset, batch_size=32, num_workers=8)

# B. Use prefetching
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, prefetch_factor=4, persistent_workers=True)

# C. Pin memory
dataloader = DataLoader(dataset, pin_memory=True)
```

---

### 2. Gradient Synchronization Overhead (DDP)

**Symptom:** backward() takes 2-3× longer in multi-GPU vs single-GPU

**Diagnosis:**
```python
# Trace shows: torch.distributed.all_reduce takes 60% of backward time
```

**Solutions:**
```python
# A. Enable gradient_as_bucket_view (saves memory copy)
from accelerate.utils import DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(gradient_as_bucket_view=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

# B. Increase DDP bucket size (fewer all-reduce calls)
ddp_kwargs = DistributedDataParallelKwargs(bucket_cap_mb=100)  # Default: 25 MB

# C. Use faster interconnect (InfiniBand > Ethernet)
```

---

### 3. Mixed Precision Overhead

**Symptom:** FP16 training slower than FP32

**Diagnosis:**
```python
# Trace shows: scaler.scale/unscale operations dominate
```

**Solutions:**
```python
# A. Reduce scaler update frequency
from accelerate.utils import GradScalerKwargs

scaler_kwargs = GradScalerKwargs(growth_interval=4000)  # Default: 2000
accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[scaler_kwargs])

# B. Use BF16 instead (no scaling overhead)
accelerator = Accelerator(mixed_precision="bf16")
```

---

### 4. Small Batch Size (GPU Underutilization)

**Symptom:** GPU SM (streaming multiprocessor) occupancy < 50%

**Diagnosis:**
```python
# Trace shows: Kernels have low occupancy
# Example: matmul only uses 30% of GPU
```

**Solutions:**
```python
# A. Increase batch size
dataloader = DataLoader(dataset, batch_size=64)  # Was: 32

# B. Use gradient accumulation
accelerator = Accelerator(gradient_accumulation_steps=4)
# Effective batch size: 32 × 4 = 128

# C. Enable micro-batching (transformer models)
```

---

## Analyzing Chrome Traces

### Opening Traces
1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load" and select trace file (e.g., `./traces/rank0_trace.json`)

### Key Metrics to Look For

| **Metric** | **Good** | **Bad** | **Fix** |
|------------|----------|---------|---------|
| GPU Utilization | > 80% | < 50% | Increase batch size, reduce data loading time |
| Data Loading Time | < 10% of iteration | > 30% | Increase num_workers, use prefetching |
| All-Reduce Time | < 15% of backward | > 30% | Use faster interconnect, increase bucket size |
| CPU-GPU Transfer | < 5% | > 15% | Pin memory, use non_blocking transfers |

---

## Memory Profiling

```python
from accelerate.utils import ProfileKwargs

# Enable memory profiling
profile_kwargs = ProfileKwargs(
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
    output_trace_dir="./memory_traces"
)

accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

with accelerator.profile() as prof:
    for batch in dataloader:
        outputs = model(batch)
        loss = loss_fn(outputs, labels)
        accelerator.backward(loss)
        prof.step()

# Analyze memory usage
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
```

**Output Example:**
```
---------------------------------  ------------  ------------  ------------
Name                               CPU Mem      CUDA Mem      Self CUDA Mem
---------------------------------  ------------  ------------  ------------
Linear                             0 b          2.00 Gb       2.00 Gb
LayerNorm                          0 b          512.00 Mb     512.00 Mb
Dropout                            0 b          256.00 Mb     256.00 Mb
```

---

## Distributed Profiling Tips

### 1. Profile on Single Process First
```python
# Don't profile all 8 GPUs at once (trace files too large)
if accelerator.process_index == 0:
    with accelerator.profile():
        train_step()
else:
    train_step()
```

### 2. Synchronize Before Profiling
```python
accelerator.wait_for_everyone()  # Ensure all processes start together

with accelerator.profile():
    train_step()
```

### 3. Profile Representative Workload
```python
# Skip warmup iterations (first iteration is always slow)
for i, batch in enumerate(dataloader):
    if i < 3:
        # Warmup
        train_step(batch)
        continue

    if i < 10:
        # Profile iterations 3-9
        with accelerator.profile():
            train_step(batch)
            prof.step()
```

---

## Quick Diagnostic Checklist

```python
# Run this to identify bottlenecks
import torch.utils.benchmark as benchmark

# 1. Measure data loading time
t = benchmark.Timer(stmt='next(iter(dataloader))', globals=globals())
print(f"Data loading: {t.timeit(100).mean * 1000:.2f} ms")

# 2. Measure forward pass
t = benchmark.Timer(stmt='model(batch)', globals=globals())
print(f"Forward pass: {t.timeit(100).mean * 1000:.2f} ms")

# 3. Measure backward pass
t = benchmark.Timer(stmt='loss.backward()', setup='loss = model(batch).sum()', globals=globals())
print(f"Backward pass: {t.timeit(100).mean * 1000:.2f} ms")

# 4. Measure optimizer step
t = benchmark.Timer(stmt='optimizer.step()', globals=globals())
print(f"Optimizer step: {t.timeit(100).mean * 1000:.2f} ms")
```

**Expected Ratios (for well-optimized training):**
- Data loading: 5-10% of iteration time
- Forward pass: 30-40%
- Backward pass: 40-50%
- Optimizer step: 5-10%

---

## Related Files
- **ProfileKwargs:** `src/accelerate/utils/dataclasses.py`
- **Profile context manager:** `src/accelerate/accelerator.py:4077`
- **PyTorch profiler docs:** https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
