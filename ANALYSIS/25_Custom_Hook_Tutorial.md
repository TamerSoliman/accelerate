# Integration 14: Custom Hook Tutorial

## Creating Custom Hooks

```python
from accelerate.hooks import ModelHook

class LoggingHook(ModelHook):
    """Logs input/output shapes"""

    def pre_forward(self, module, *args, **kwargs):
        print(f"Input shape: {args[0].shape}")
        return args, kwargs

    def post_forward(self, module, output):
        print(f"Output shape: {output.shape}")
        return output

# Usage
from accelerate.hooks import add_hook_to_module

add_hook_to_module(model.layer, LoggingHook())
```

---

## Gradient Clipping Hook

```python
class GradientClippingHook(ModelHook):
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def post_forward(self, module, output):
        # Clip gradients during backward
        if output.requires_grad:
            output.register_hook(
                lambda grad: torch.nn.utils.clip_grad_norm_(
                    module.parameters(), self.max_norm
                )
            )
        return output
```

---

## Memory Tracking Hook

```python
class MemoryTrackingHook(ModelHook):
    def pre_forward(self, module, *args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        return args, kwargs

    def post_forward(self, module, output):
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"{module.__class__.__name__}: {mem:.2f}GB")
        return output
```

---

Next: **26_CPU_Offload_Strategies.md**
