# Integration 16: Hook Performance Analysis

## Overhead Measurement

```python
import time

class TimingHook(ModelHook):
    def __init__(self):
        self.times = []

    def pre_forward(self, module, *args, **kwargs):
        self.start = time.time()
        return args, kwargs

    def post_forward(self, module, output):
        elapsed = time.time() - self.start
        self.times.append(elapsed)
        print(f"Layer time: {elapsed*1000:.2f}ms")
        return output
```

---

## Memory vs Speed Trade-offs

| Hook Type | Memory Saved | Speed Penalty | Overhead |
|-----------|--------------|---------------|----------|
| AlignDevicesHook | 0% | ~1% | Minimal |
| CpuOffload | 90%+ | 10-50× | High (CPU↔GPU transfer) |
| LayerwiseCasting | 50% | ~5% | Low |

---

## Optimization Tips

1. **Minimize CPU↔GPU transfers** - Batch operations
2. **Use pinned memory** - Faster transfers
3. **Overlap computation** - Transfer next layer while executing current
4. **Profile first** - Identify bottlenecks before optimizing

---

Next: **Group 5 - Model Utilities**
