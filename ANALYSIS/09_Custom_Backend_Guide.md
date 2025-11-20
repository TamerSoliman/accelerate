# Enhancement 3: Custom Backend Implementation Guide

## Overview
This guide explains how to add a custom distributed training backend to Accelerate, following the patterns established by DeepSpeed, FSDP, and Megatron-LM integrations.

---

## Architecture: Plugin System

### Core Components
1. **Plugin Class** - Configuration object (e.g., `MyBackendPlugin`)
2. **DistributedType** - Enum value for backend identification
3. **Preparation Method** - `_prepare_mybackend()` in Accelerator
4. **State Integration** - Add to `AcceleratorState`
5. **Wrapper Classes** - Model/optimizer/scheduler wrappers

---

## Step 1: Create Plugin Class

**Location:** `src/accelerate/utils/dataclasses.py`

```python
@dataclass
class MyBackendPlugin:
    """
    Plugin for integrating MyBackend distributed training framework.

    Args:
        config_file: Path to MyBackend configuration JSON
        sharding_strategy: How to shard parameters ("full" or "partial")
        offload_to_cpu: Whether to offload parameters to CPU
    """

    config_file: str | None = None
    sharding_strategy: Literal["full", "partial"] = "full"
    offload_to_cpu: bool = False

    def __post_init__(self):
        # Initialize from config file if provided
        if self.config_file is not None:
            with open(self.config_file) as f:
                config = json.load(f)
                self.sharding_strategy = config.get("sharding_strategy", self.sharding_strategy)
                self.offload_to_cpu = config.get("offload_to_cpu", self.offload_to_cpu)

        # Validate configuration
        if self.sharding_strategy not in ["full", "partial"]:
            raise ValueError(f"Invalid sharding_strategy: {self.sharding_strategy}")

    def to_dict(self):
        """Convert plugin to dictionary for serialization"""
        return {
            "config_file": self.config_file,
            "sharding_strategy": self.sharding_strategy,
            "offload_to_cpu": self.offload_to_cpu,
        }
```

---

## Step 2: Add DistributedType Enum

**Location:** `src/accelerate/utils/dataclasses.py:DistributedType`

```python
class DistributedType(str, enum.Enum):
    # Existing types...
    NO = "NO"
    MULTI_GPU = "MULTI_GPU"
    DEEPSPEED = "DEEPSPEED"
    FSDP = "FSDP"
    MEGATRON_LM = "MEGATRON_LM"

    # Add your backend
    MYBACKEND = "MYBACKEND"
```

---

## Step 3: Integrate with AcceleratorState

**Location:** `src/accelerate/state.py:AcceleratorState.__init__`

```python
def __init__(
    self,
    # ... existing args ...
    mybackend_plugin: MyBackendPlugin | None = None,
    **kwargs
):
    # ... existing initialization ...

    # Check for MyBackend
    if mybackend_plugin is None:
        mybackend_plugin = (
            MyBackendPlugin()
            if os.environ.get("ACCELERATE_USE_MYBACKEND", "false").lower() == "true"
            else None
        )
    else:
        os.environ["ACCELERATE_USE_MYBACKEND"] = "true"

    self.mybackend_plugin = mybackend_plugin

    # Set distributed type
    if self.mybackend_plugin is not None:
        self.distributed_type = DistributedType.MYBACKEND
```

---

## Step 4: Create Wrapper Classes

**Location:** `src/accelerate/utils/mybackend.py` (new file)

```python
# Model Wrapper
class MyBackendModelWrapper(torch.nn.Module):
    """Wraps model with MyBackend's distributed training logic"""

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

        # Initialize MyBackend's distributed setup
        import mybackend
        self.backend_model = mybackend.distributed_model(
            model,
            sharding_strategy=config.sharding_strategy,
            offload_to_cpu=config.offload_to_cpu,
        )

    def forward(self, *args, **kwargs):
        return self.backend_model(*args, **kwargs)

    def state_dict(self):
        # Return full model state_dict (gather shards if needed)
        return self.backend_model.state_dict()

    def load_state_dict(self, state_dict):
        self.backend_model.load_state_dict(state_dict)


# Optimizer Wrapper
class MyBackendOptimizerWrapper:
    """Wraps optimizer for MyBackend compatibility"""

    def __init__(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model

        # Integrate with backend
        import mybackend
        self.backend_optimizer = mybackend.distributed_optimizer(
            optimizer, model
        )

    def step(self, closure=None):
        return self.backend_optimizer.step(closure)

    def zero_grad(self):
        return self.backend_optimizer.zero_grad()

    def state_dict(self):
        return self.backend_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.backend_optimizer.load_state_dict(state_dict)
```

---

## Step 5: Add Preparation Method

**Location:** `src/accelerate/accelerator.py`

```python
def _prepare_mybackend(self, *args):
    """
    Prepares models, optimizers, schedulers for MyBackend.

    MyBackend requires:
    1. Model and optimizer prepared together
    2. Custom initialization for sharded parameters
    3. Special handling for checkpointing
    """
    from .utils.mybackend import MyBackendModelWrapper, MyBackendOptimizerWrapper

    # Extract components
    model = None
    optimizer = None
    scheduler = None
    dataloaders = []

    for obj in args:
        if isinstance(obj, torch.nn.Module):
            model = obj
        elif isinstance(obj, torch.optim.Optimizer):
            optimizer = obj
        elif isinstance(obj, LRScheduler):
            scheduler = obj
        elif isinstance(obj, torch.utils.data.DataLoader):
            dataloaders.append(obj)

    if model is None:
        raise ValueError("MyBackend requires a model")

    # Wrap model with MyBackend
    model = MyBackendModelWrapper(model, self.state.mybackend_plugin)
    self._models.append(model)

    # Wrap optimizer
    if optimizer is not None:
        optimizer = MyBackendOptimizerWrapper(optimizer, model)
        self._optimizers.append(optimizer)

    # Prepare schedulers (standard approach)
    if scheduler is not None:
        scheduler = self._prepare_one(scheduler, first_pass=False)

    # Prepare dataloaders (standard approach)
    prepared_dataloaders = []
    for dataloader in dataloaders:
        prepared_dataloaders.append(
            self.prepare_data_loader(dataloader, device_placement=True)
        )

    # Return in original order
    result = []
    for obj in args:
        if isinstance(obj, torch.nn.Module):
            result.append(model)
        elif isinstance(obj, torch.optim.Optimizer):
            result.append(optimizer)
        elif isinstance(obj, LRScheduler):
            result.append(scheduler)
        elif isinstance(obj, torch.utils.data.DataLoader):
            result.append(prepared_dataloaders.pop(0))
        else:
            result.append(obj)

    return tuple(result)
```

---

## Step 6: Route to Backend in prepare()

**Location:** `src/accelerate/accelerator.py:prepare()`

```python
def prepare(self, *args, device_placement=None):
    # ... validation ...

    if self.distributed_type == DistributedType.MYBACKEND:
        result = self._prepare_mybackend(*args)
    elif self.distributed_type == DistributedType.DEEPSPEED:
        result = self._prepare_deepspeed(*args)
    # ... other backends ...
    else:
        # Standard preparation
        result = tuple(
            self._prepare_one(obj, first_pass=True, device_placement=d)
            for obj, d in zip(args, device_placement)
        )

    return result if len(result) > 1 else result[0]
```

---

## Step 7: Handle Checkpointing

**Location:** `src/accelerate/accelerator.py:save_state()`

```python
def save_state(self, output_dir, safe_serialization=True, **kwargs):
    # ...

    for i, model in enumerate(self._models):
        if self.distributed_type == DistributedType.MYBACKEND:
            # MyBackend-specific save
            logger.info("Saving MyBackend model")
            model.backend_model.save_checkpoint(output_dir, f"model_{i}")
        elif self.distributed_type == DistributedType.DEEPSPEED:
            # DeepSpeed save
            model.save_checkpoint(output_dir, f"model_{i}")
        # ... other backends ...
```

---

## Step 8: Add CLI Support

**Location:** `src/accelerate/commands/config/config_args.py`

```python
# Add to ClusterConfig dataclass
@dataclass
class ClusterConfig(BaseConfig):
    # ... existing fields ...

    use_mybackend: bool = field(
        default=False,
        metadata={"help": "Whether to use MyBackend for distributed training"}
    )
    mybackend_config_file: str = field(
        default=None,
        metadata={"help": "Path to MyBackend configuration file"}
    )

    def __post_init__(self):
        # ... existing logic ...

        if self.use_mybackend:
            self.distributed_type = DistributedType.MYBACKEND
```

---

## Step 9: Add Tests

**Location:** `tests/mybackend/test_mybackend.py` (new file)

```python
import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import MyBackendPlugin


class TestMyBackend:
    def test_basic_training(self):
        """Test basic training with MyBackend"""
        plugin = MyBackendPlugin(sharding_strategy="full")
        accelerator = Accelerator(mybackend_plugin=plugin)

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        model, optimizer = accelerator.prepare(model, optimizer)

        # Forward pass
        x = torch.randn(4, 10, device=accelerator.device)
        y = model(x)
        loss = y.sum()

        # Backward pass
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        # Verify model was wrapped correctly
        assert isinstance(model, MyBackendModelWrapper)

    def test_checkpoint_save_load(self):
        """Test checkpointing with MyBackend"""
        plugin = MyBackendPlugin()
        accelerator = Accelerator(mybackend_plugin=plugin)

        model, optimizer = accelerator.prepare(model, optimizer)

        # Save
        accelerator.save_state("checkpoint")

        # Load
        accelerator.load_state("checkpoint")
```

---

## Complete Example: Minimal Backend Integration

```python
# 1. Plugin definition (dataclasses.py)
@dataclass
class MinimalBackendPlugin:
    enabled: bool = True

# 2. Add to DistributedType
class DistributedType(str, enum.Enum):
    MINIMAL_BACKEND = "MINIMAL_BACKEND"

# 3. Preparation method (accelerator.py)
def _prepare_minimal_backend(self, *args):
    # Wrap model with custom logic
    model = args[0]
    wrapped_model = MyCustomWrapper(model)
    return (wrapped_model,) + args[1:]

# 4. Route in prepare()
if self.distributed_type == DistributedType.MINIMAL_BACKEND:
    result = self._prepare_minimal_backend(*args)

# 5. Usage
plugin = MinimalBackendPlugin()
accelerator = Accelerator(minimal_backend_plugin=plugin)
model = accelerator.prepare(model)
```

---

## Best Practices

### 1. Follow Existing Patterns
- Study DeepSpeed, FSDP implementations
- Use same naming conventions (`prepare_X`, `save_X_model`)
- Maintain API consistency

### 2. Handle All Object Types
- Models (required)
- Optimizers (required)
- Schedulers (optional)
- DataLoaders (usually standard)

### 3. Support Checkpointing
- Implement `state_dict()` and `load_state_dict()`
- Handle sharded state if applicable
- Test save/load cycles

### 4. Add Comprehensive Tests
- Basic training loop
- Multi-GPU scenarios
- Checkpointing
- Edge cases (tied weights, etc.)

### 5. Document Limitations
- What features are supported/unsupported
- Hardware requirements
- Performance characteristics

---

## Related Files
- **Plugin examples:** `src/accelerate/utils/dataclasses.py:1086` (DeepSpeedPlugin)
- **Preparation examples:** `src/accelerate/accelerator.py:2064` (_prepare_deepspeed)
- **Wrapper examples:** `src/accelerate/utils/deepspeed.py`
- **State integration:** `src/accelerate/state.py:AcceleratorState`
