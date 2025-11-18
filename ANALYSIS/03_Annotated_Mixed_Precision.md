# Annotated: Mixed Precision (AMP) Handling in Accelerate

## Overview
Automatic Mixed Precision (AMP) training reduces memory usage and increases training speed by using lower-precision (FP16/BF16/FP8) for most operations while maintaining FP32 for critical operations. Accelerate handles the complexity of mixed precision automatically, including loss scaling, gradient unscaling, and numerical stability.

---

## Core Components of Mixed Precision

### 1. GradScaler Initialization
**File:** `src/accelerate/accelerator.py:562-612`

```python
# ============================================================================
# GRADSCALER INITIALIZATION (FP16 Mixed Precision)
# ============================================================================
# **WHAT:** Create a GradScaler for loss scaling during FP16 training
# **HOW:** Instantiate torch.cuda.amp.GradScaler or ShardedGradScaler (for FSDP)
# **WHY:** FP16 has limited dynamic range - small gradients underflow to zero.
#          Loss scaling multiplies loss by a large factor before backward(),
#          then unscales gradients before optimizer.step()

self.scaler = None
self.native_amp = False

# Check if FP16 mixed precision is enabled
if (
    self.state.mixed_precision == "fp16"
    and self.device.type != "cpu"
    and self.distributed_type not in (DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM)
):
    self.native_amp = True
    supported_device = ("xpu", "cuda", "npu", "xla", "mlu", "musa", "hpu", "sdaa", "mps")

    # Validate device supports FP16
    if self.device.type not in supported_device or is_torch_xla_available(check_is_tpu=True):
        raise ValueError(
            f"fp16 mixed precision requires a device in {supported_device} (not {self.device.type!r})."
        )

    # Get scaler configuration from kwargs handler
    kwargs = self.scaler_handler.to_kwargs() if self.scaler_handler is not None else {}

    # **CRITICAL BRANCH:** FSDP2 uses a different scaler
    if self.is_fsdp2:
        # FSDP2 requires ShardedGradScaler for distributed loss scaling
        self.scaler = get_fsdp2_grad_scaler(device=self.device.type, **kwargs)
    else:
        # Standard GradScaler for DDP or single-GPU
        self.scaler = get_grad_scaler(self.distributed_type, **kwargs)
```

**Key Insight:** The scaler is backend-specific:
- **DDP/Single-GPU:** `torch.cuda.amp.GradScaler`
- **FSDP2:** Custom `ShardedGradScaler` that coordinates scaling across shards
- **DeepSpeed/Megatron:** No scaler needed (handled internally)

---

### 2. Loss Scaling in backward()
**File:** `src/accelerate/accelerator.py:2708-2740`

```python
def backward(self, loss, **kwargs):
    """
    Performs backward pass with automatic mixed precision support.

    **WHAT:** Compute gradients with proper scaling for mixed precision
    **HOW:** Routes to backend-specific backward implementations
    **WHY:** Different backends require different backward pass handling
    """

    # ========================================================================
    # GRADIENT ACCUMULATION SCALING
    # ========================================================================
    # **WHAT:** Divide loss by gradient_accumulation_steps
    # **HOW:** Scale down the loss before backward pass
    # **WHY:** With gradient accumulation, we sum gradients over N steps.
    #          Dividing loss by N gives us the mean gradient, not the sum.
    #          This ensures consistent learning rate behavior.

    if self.distributed_type != DistributedType.DEEPSPEED:
        # DeepSpeed handles this internally
        loss = loss / self.gradient_accumulation_steps

    # ========================================================================
    # BACKEND-SPECIFIC BACKWARD DISPATCH
    # ========================================================================

    if self.distributed_type == DistributedType.DEEPSPEED:
        # DeepSpeed has its own backward implementation with built-in scaling
        self.deepspeed_engine_wrapped.backward(loss, sync_gradients=self.sync_gradients, **kwargs)

    elif self.distributed_type == DistributedType.MEGATRON_LM:
        # Megatron-LM handles backward internally
        return

    elif self.scaler is not None:
        # ====================================================================
        # **FP16 MIXED PRECISION PATH** (Most Common)
        # ====================================================================
        # **WHAT:** Scale loss before backward to prevent gradient underflow
        # **HOW:** self.scaler.scale(loss) multiplies loss by scale_factor
        # **WHY:** FP16 can only represent values down to ~6e-8. Small gradients
        #          would underflow to zero. Scaling prevents this.

        # **CRITICAL LINE:** This is where loss scaling happens
        self.scaler.scale(loss).backward(**kwargs)

        # **What happens internally:**
        # 1. scaler.scale(loss) → loss * scaler._scale (typically 2^16 initially)
        # 2. scaled_loss.backward() → gradients are also scaled by scaler._scale
        # 3. Later, scaler.unscale_() divides gradients by scaler._scale
        # 4. scaler.step() checks for inf/NaN, adjusts scale, then calls optimizer.step()

    elif learning_rate is not None and self.has_lomo_optimizer:
        # LOMO (Low-Memory Optimization) uses a custom backward
        self.lomo_backward(loss, learning_rate)

    else:
        # ====================================================================
        # **STANDARD BACKWARD** (BF16, FP32, or no mixed precision)
        # ====================================================================
        # **WHAT:** Standard PyTorch backward pass
        # **HOW:** No scaling needed for BF16 (larger dynamic range than FP16)
        # **WHY:** BF16 has the same exponent range as FP32, so no underflow risk

        loss.backward(**kwargs)
```

---

### 3. Gradient Unscaling
**File:** `src/accelerate/accelerator.py:2801-2834`

```python
def unscale_gradients(self, optimizer=None):
    """
    Unscale the gradients in mixed precision training with AMP.

    **WHAT:** Divide gradients by the scaler's scale factor
    **HOW:** Call self.scaler.unscale_(optimizer)
    **WHY:** After backward(), gradients are scaled. We must unscale before:
             - Gradient clipping (must clip true gradient magnitude)
             - Optimizer step (must update with true gradients)
    """

    # Only unscale for FP16 native AMP
    if self.native_amp and self.mixed_precision == "fp16":
        # ====================================================================
        # OPTIMIZER SELECTION
        # ====================================================================
        # **WHAT:** Determine which optimizers to unscale
        # **HOW:** Use passed optimizer or all prepared optimizers
        # **WHY:** Multiple optimizers may exist (e.g., separate for generator/discriminator)

        if optimizer is None:
            # Unscale all optimizers
            optimizer = self._optimizers
        elif not isinstance(optimizer, (tuple, list)):
            optimizer = [optimizer]

        # ====================================================================
        # UNWRAP AND UNSCALE
        # ====================================================================
        # **WHAT:** Unscale gradients for each optimizer
        # **HOW:** Unwrap AcceleratedOptimizer wrapper, then call scaler.unscale_()
        # **WHY:** scaler.unscale_() modifies gradients in-place

        for opt in optimizer:
            # Unwrap AcceleratedOptimizer to get the underlying optimizer
            while isinstance(opt, AcceleratedOptimizer):
                opt = opt.optimizer

            # **CRITICAL CALL:** Unscale gradients
            self.scaler.unscale_(opt)

            # **What happens internally:**
            # 1. For each parameter group in optimizer:
            # 2.   For each parameter with gradients:
            # 3.     grad = grad / scaler._scale
            # 4. Checks for inf/NaN in gradients
            # 5. Sets scaler._found_inf flag if overflow detected
```

**Key Insight:** Unscaling is idempotent - calling it multiple times has no effect. This is safe because `scaler.unscale_()` tracks whether it has already been called for a given optimizer.

---

### 4. Gradient Clipping with Unscaling
**File:** `src/accelerate/accelerator.py:2836-2897`

```python
def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
    """
    Should be used in place of `torch.nn.utils.clip_grad_norm_`.

    **WHAT:** Clip gradient norm to prevent exploding gradients
    **HOW:** Unscale gradients first, then apply standard gradient clipping
    **WHY:** Clipping must be done on true gradients, not scaled gradients
    """

    # ========================================================================
    # FSDP SPECIAL HANDLING
    # ========================================================================
    # **WHAT:** For FSDP, use model.clip_grad_norm_() instead of torch utility
    # **HOW:** Match parameters to the correct FSDP model, use its method
    # **WHY:** FSDP shards parameters across GPUs. Regular clipping would only
    #          clip local shards, not the full gradient norm. FSDP's method
    #          computes the global norm correctly.

    if self.distributed_type == DistributedType.FSDP:
        # Unscale before clipping
        self.unscale_gradients()

        parameters = [p for p in parameters]
        for model in self._models:
            if parameters == [p for p in model.parameters()]:
                if not self.is_fsdp2:
                    # FSDP1 has built-in clip_grad_norm_
                    return model.clip_grad_norm_(max_norm, norm_type)
                else:
                    # FSDP2 uses standard PyTorch clipping
                    return torch.nn.utils.clip_grad_norm_(
                        parameters, max_norm, norm_type=norm_type
                    )

    # ========================================================================
    # DEEPSPEED SPECIAL HANDLING
    # ========================================================================
    # **WHAT:** DeepSpeed handles clipping internally
    # **HOW:** Retrieve the global gradient norm from DeepSpeed engine
    # **WHY:** DeepSpeed manages gradients internally, so we just query the norm

    elif self.distributed_type == DistributedType.DEEPSPEED:
        if self.deepspeed_engine_wrapped is not None:
            return self.deepspeed_engine_wrapped.get_global_grad_norm()
        return None

    # ========================================================================
    # XLA (TPU) SPECIAL HANDLING
    # ========================================================================
    # **WHAT:** For TPU, synchronize gradients across devices before clipping
    # **HOW:** Use xm.all_reduce to sum gradients across all TPU cores
    # **WHY:** TPU uses lazy evaluation - gradients aren't synchronized until
    #          explicitly requested. We need global gradients for correct clipping.

    elif self.distributed_type == DistributedType.XLA:
        for acc_opt in self._optimizers:
            if not acc_opt.gradient_state.is_xla_gradients_synced:
                opt = acc_opt
                while isinstance(opt, AcceleratedOptimizer):
                    opt = opt.optimizer
                # Fetch gradients and reduce across TPU cores
                gradients = xm._fetch_gradients(opt)
                xm.all_reduce("sum", gradients, scale=1.0 / self.num_processes)
                acc_opt.gradient_state.is_xla_gradients_synced = True

        # Handle FSDP on XLA
        if os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true":
            self.unscale_gradients()
            parameters = [p for p in parameters]
            for model in self._models:
                if parameters == [p for p in model.parameters()]:
                    return model.clip_grad_norm_(max_norm, norm_type)

    # ========================================================================
    # STANDARD PATH (DDP, Single GPU)
    # ========================================================================
    # **WHAT:** Unscale then clip using standard PyTorch function
    # **HOW:** Call unscale_gradients(), then torch.nn.utils.clip_grad_norm_()
    # **WHY:** Standard approach works for DDP since gradients are already
    #          synchronized during backward pass

    self.unscale_gradients()
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)
```

**Critical Flow:**
```
User calls: accelerator.clip_grad_norm_(model.parameters(), 1.0)
    ↓
1. Unscale gradients: grad = grad / scale_factor
2. Compute global gradient norm: ||grad||
3. If ||grad|| > max_norm:
     grad = grad * (max_norm / ||grad||)
```

---

### 5. Optimizer Step with Scaler
**File:** `src/accelerate/optimizer.py:145-200`

```python
class AcceleratedOptimizer(torch.optim.Optimizer):
    """
    Wrapper around torch optimizer that handles:
    - Gradient accumulation (only step when sync_gradients=True)
    - Gradient scaling (for FP16 mixed precision)
    - XLA gradient synchronization
    """

    def __init__(self, optimizer, device_placement=True, scaler=None):
        self.optimizer = optimizer
        self.scaler = scaler
        self.accelerator_state = AcceleratorState()
        self.gradient_state = GradientState()
        self.device_placement = device_placement
        self._is_overflow = False

        # ====================================================================
        # SCALER STEP PATCHING
        # ====================================================================
        # **WHAT:** Replace optimizer.step() with a patched version
        # **HOW:** Save original step method, create patched version with scaler
        # **WHY:** We need to intercept optimizer.step() to call scaler.step()
        #          instead of directly updating parameters

        if self.scaler is not None:
            self._accelerate_step_called = False
            self._optimizer_original_step_method = self.optimizer.step
            self._optimizer_patched_step_method = patch_optimizer_step(self, self.optimizer.step)

    def step(self, closure=None):
        """
        Performs optimizer step with gradient accumulation and scaling support.
        """

        # ====================================================================
        # XLA GRADIENT SYNCHRONIZATION
        # ====================================================================
        # **WHAT:** For TPU, synchronize gradients before stepping
        # **HOW:** Use xm.optimizer_step() which handles gradient reduction
        # **WHY:** TPU uses lazy evaluation - must explicitly sync gradients

        if (
            not self.gradient_state.is_xla_gradients_synced
            and self.accelerator_state.distributed_type == DistributedType.XLA
            and self.gradient_state.sync_gradients
        ):
            # Sync gradients and step
            xm.optimizer_step(self.optimizer, optimizer_args={"closure": closure})
            self.gradient_state.is_xla_gradients_synced = True
            return

        # ====================================================================
        # GRADIENT ACCUMULATION GATING
        # ====================================================================
        # **WHAT:** Only step when gradients should be synchronized
        # **HOW:** Check self.gradient_state.sync_gradients flag
        # **WHY:** During gradient accumulation, we accumulate for N steps,
        #          then sync and step on the Nth step

        if self.gradient_state.sync_gradients:
            # ================================================================
            # SCALER STEP (FP16 Mixed Precision)
            # ================================================================
            # **WHAT:** Step optimizer with gradient scaling
            # **HOW:** self.scaler.step() unscales gradients, checks for inf/NaN,
            #          and calls optimizer.step() if no overflow
            # **WHY:** Must check for gradient overflow before updating parameters

            if self.scaler is not None:
                # **CRITICAL PATH:** Scaler-aware step

                # Scale factor before step (for debugging)
                scale_before = self.scaler.get_scale()

                # **KEY CALL:** scaler.step(optimizer)
                # What happens:
                # 1. If gradients haven't been unscaled yet, unscale them
                # 2. Check for inf/NaN in gradients (sets _found_inf flag)
                # 3. If _found_inf is False:
                #      - Call optimizer.step() to update parameters
                #      - Increase scale_factor (e.g., scale *= 2)
                # 4. If _found_inf is True:
                #      - Skip optimizer.step() (parameters unchanged)
                #      - Decrease scale_factor (e.g., scale /= 2)
                #      - Zero gradients
                self.scaler.step(self.optimizer, closure)

                # Check if step was skipped due to overflow
                if self.scaler.get_scale() < scale_before:
                    self._is_overflow = True

                # **UPDATE SCALE:** Adjust scale_factor for next iteration
                # - If no overflow: increase scale (up to max)
                # - If overflow: decrease scale
                # This creates a dynamic scaling that adapts to the training dynamics
                self.scaler.update()

            else:
                # ============================================================
                # STANDARD STEP (BF16, FP32, or no mixed precision)
                # ============================================================
                # **WHAT:** Regular optimizer step without scaling
                # **HOW:** Directly call self.optimizer.step()
                # **WHY:** BF16/FP32 don't need gradient scaling

                self.optimizer.step(closure)
```

**Scaler Step Internals:**
```python
# Inside torch.cuda.amp.GradScaler.step(optimizer):
def step(self, optimizer, *args, **kwargs):
    # 1. Unscale gradients (if not already done)
    if not self._per_optimizer_states[id(optimizer)]["stage"] == OptState.UNSCALED:
        self.unscale_(optimizer)

    # 2. Check for overflow (inf/NaN in gradients)
    found_inf = self._found_inf_per_device(self._check_inf_per_device(optimizer))

    # 3. Step optimizer only if no overflow
    if not found_inf:
        retval = optimizer.step(*args, **kwargs)
    else:
        retval = None  # Step skipped

    return retval

# Inside torch.cuda.amp.GradScaler.update():
def update(self, new_scale=None):
    if new_scale is not None:
        self._scale = new_scale
    else:
        if self._found_inf:
            # Overflow detected: reduce scale
            self._scale = self._scale * self._backoff_factor  # Default: scale /= 2
            self._growth_tracker = 0
        else:
            # No overflow: potentially increase scale
            self._growth_tracker += 1
            if self._growth_tracker == self._growth_interval:  # Default: 2000 steps
                self._scale = self._scale * self._growth_factor  # Default: scale *= 2
                self._growth_tracker = 0
```

---

## Complete Training Loop Flow with Mixed Precision

```python
# User Code
accelerator = Accelerator(mixed_precision="fp16")
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    # ========================================================================
    # 1. FORWARD PASS (with autocast)
    # ========================================================================
    # Accelerate automatically wraps forward pass with torch.autocast()
    # - Linear layers computed in FP16
    # - Loss functions computed in FP32
    # - Reductions (sum, mean) computed in FP32

    outputs = model(batch)  # Executed in mixed precision context
    loss = loss_fn(outputs, labels)  # Loss is FP32

    # ========================================================================
    # 2. BACKWARD PASS (with gradient scaling)
    # ========================================================================
    # accelerator.backward(loss) does:
    # - Divides loss by gradient_accumulation_steps
    # - Calls scaler.scale(loss).backward()
    #   - loss is scaled by 2^16 (initially)
    #   - Gradients are computed in FP16, but scaled

    accelerator.backward(loss)

    # ========================================================================
    # 3. GRADIENT CLIPPING (with unscaling)
    # ========================================================================
    # accelerator.clip_grad_norm_() does:
    # - Calls scaler.unscale_(optimizer)
    #   - Divides all gradients by scale_factor (2^16)
    #   - Now gradients are true FP32 gradients
    # - Computes ||grad|| and clips if necessary

    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(model.parameters(), 1.0)

    # ========================================================================
    # 4. OPTIMIZER STEP (with overflow detection)
    # ========================================================================
    # optimizer.step() does:
    # - Checks gradients for inf/NaN
    # - If no overflow:
    #     - Updates parameters: param = param - lr * grad
    #     - Increases scale_factor for next iteration
    # - If overflow:
    #     - Skips parameter update
    #     - Decreases scale_factor
    #     - Zeros gradients
    # - Calls scaler.update() to adjust scale_factor

    optimizer.step()

    # ========================================================================
    # 5. GRADIENT ZEROING
    # ========================================================================
    # optimizer.zero_grad() respects gradient accumulation
    # - Only zeros if sync_gradients=True

    optimizer.zero_grad()
```

---

## Autocast Context Management

**File:** `src/accelerate/utils/__init__.py` (search for `get_mixed_precision_context_manager`)

```python
def get_mixed_precision_context_manager(native_amp, autocast_kwargs):
    """
    Returns the proper torch.autocast context manager for mixed precision.

    **WHAT:** Create context manager for automatic mixed precision
    **HOW:** Return torch.autocast() or torch.cuda.amp.autocast()
    **WHY:** Different PyTorch versions have different autocast APIs
    """

    if native_amp:
        # Use device-specific autocast
        if is_torch_version(">=", "1.10"):
            # Modern PyTorch (1.10+) has generic torch.autocast()
            return torch.autocast(device_type=autocast_kwargs.pop("device_type"), **autocast_kwargs)
        else:
            # Legacy PyTorch (< 1.10) uses torch.cuda.amp.autocast()
            return torch.cuda.amp.autocast(**autocast_kwargs)
    else:
        # No mixed precision, use null context
        return contextlib.nullcontext()
```

This context manager is applied automatically during:
- Model forward pass (in `prepare_model`)
- Custom training loops (when using `accelerator.autocast()`)

---

## Key Takeaways

### 1. The Four Critical Operations
1. **Scale Loss:** `scaler.scale(loss).backward()` - Prevents gradient underflow
2. **Unscale Gradients:** `scaler.unscale_(optimizer)` - Restore true gradient values
3. **Check Overflow:** Internal to `scaler.step()` - Detect inf/NaN gradients
4. **Update Scale:** `scaler.update()` - Dynamically adjust scale_factor

### 2. Why Dynamic Scaling?
- **Problem:** Static scale might be too large (causing overflow) or too small (losing precision)
- **Solution:** Start with large scale (2^16), decrease if overflow detected, increase if stable
- **Result:** Adaptive scaling that maximizes precision while avoiding overflow

### 3. Backend Differences
| Backend | Scaler Used | Backward Handling |
|---------|-------------|-------------------|
| DDP | `torch.cuda.amp.GradScaler` | Standard scaler.scale() |
| FSDP1 | `ShardedGradScaler` | Distributed scaling |
| FSDP2 | Custom FSDP2 scaler | Distributed scaling |
| DeepSpeed | Internal DeepSpeed scaler | Handled by DeepSpeed |
| Megatron-LM | Internal Megatron scaler | Handled by Megatron |
| Single GPU | `torch.cuda.amp.GradScaler` | Standard scaler.scale() |

### 4. BF16 vs FP16
- **FP16:** Requires GradScaler (8-bit exponent, 7-bit mantissa)
- **BF16:** No scaler needed (8-bit exponent, 7-bit mantissa, but same range as FP32)
- **FP8:** Backend-specific handling (Transformer Engine, TorchAO, MS-AMP)

---

## Related Files

1. **GradScaler initialization:** `src/accelerate/utils/__init__.py` (get_grad_scaler, get_fsdp2_grad_scaler)
2. **AcceleratedOptimizer:** `src/accelerate/optimizer.py:38`
3. **Backward implementation:** `src/accelerate/accelerator.py:2708`
4. **Gradient clipping:** `src/accelerate/accelerator.py:2836`
5. **Autocast context:** `src/accelerate/utils/__init__.py` (get_mixed_precision_context_manager)
