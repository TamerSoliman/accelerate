# Integration 15: CPU Offload Strategies

## Full Model Offload

```python
from accelerate import cpu_offload

cpu_offload(model, execution_device="cuda:0")
```

**Memory:** Minimal GPU usage
**Speed:** 10-50× slower

---

## Selective Layer Offload

```python
# Offload only large layers
for i in [20, 21, 22]:  # Specific layers
    add_hook_to_module(
        model.layers[i],
        CpuOffload(execution_device="cuda:0")
    )
```

**Best for:** Models partially fitting in GPU

---

## Sequential Offload (Pipeline)

```python
# Layer N on GPU while layer N+1 loads from CPU
for layer in model.layers:
    add_hook_to_module(layer, CpuOffload(execution_device="cuda:0"))
```

---

## Performance Comparison

| Strategy | GPU Memory | Speed | Use Case |
|----------|-----------|-------|----------|
| No offload | 100% | 1× | Standard |
| Partial offload | 50% | 2-5× | Tight memory |
| Full offload | <10% | 10-50× | Extreme memory limit |

---

Next: **27_Hook_Performance_Analysis.md**
