# Integration 13: Hook Execution Deep Dive

## Overview

Hooks modify model behavior by intercepting forward passes. Used for CPU offloading, device alignment, and mixed precision.

**File:** `src/accelerate/hooks.py:43-99` (ModelHook base class)

---

## Hook Lifecycle

```python
class ModelHook:
    def init_hook(self, module):
        # Called when hook attached to module
        return module

    def pre_forward(self, module, *args, **kwargs):
        # Called BEFORE forward pass
        # Can modify inputs
        return args, kwargs

    def post_forward(self, module, output):
        # Called AFTER forward pass
        # Can modify outputs
        return output

    def detach_hook(self, module):
        # Called when hook removed
        return module
```

---

## Execution Flow

```
1. add_hook_to_module(layer, hook)
   ↓
2. hook.init_hook(layer)  # Setup
   ↓
3. layer.forward(x)       # User calls forward
   ↓
4. hook.pre_forward(layer, x)  # Modify inputs
   ↓
5. original_forward(modified_x)  # Execute
   ↓
6. hook.post_forward(layer, output)  # Modify outputs
   ↓
7. return modified_output
```

---

## Built-in Hooks

### 1. AlignDevicesHook

```python
from accelerate.hooks import AlignDevicesHook

# Automatically moves inputs to layer's device
hook = AlignDevicesHook(execution_device="cuda:0")
add_hook_to_module(layer, hook)
```

---

### 2. CpuOffload

```python
from accelerate.hooks import CpuOffload

# Keeps weights on CPU, moves to GPU during forward
hook = CpuOffload(execution_device="cuda:0")
add_hook_to_module(layer, hook)
```

---

### 3. SequentialHook

```python
from accelerate.hooks import SequentialHook

# Combine multiple hooks
hook = SequentialHook(AlignDevicesHook(), CpuOffload())
```

---

## File References

- **ModelHook**: `src/accelerate/hooks.py:43-99`
- **AlignDevicesHook**: `src/accelerate/hooks.py` (~line 130+)
- **CpuOffload**: `src/accelerate/hooks.py` (~line 200+)

---

Next: **25_Custom_Hook_Tutorial.md**
