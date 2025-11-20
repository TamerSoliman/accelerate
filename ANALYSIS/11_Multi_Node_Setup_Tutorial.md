# Enhancement 5: Multi-Node Setup Tutorial

## Overview
This tutorial provides step-by-step instructions for setting up distributed training across multiple machines (nodes) using Accelerate.

---

## Prerequisites

- **2+ machines** with identical software environment
- **Network connectivity** between all machines
- **Shared storage** (NFS/Lustre) or synchronized code on all nodes
- **Same PyTorch version** on all machines
- **Same Python environment** (packages, versions)

---

## Architecture: Multi-Node Training

```
Node 0 (Main Process)                 Node 1
├── GPU 0 (Rank 0)                   ├── GPU 0 (Rank 4)
├── GPU 1 (Rank 1)                   ├── GPU 1 (Rank 5)
├── GPU 2 (Rank 2)                   ├── GPU 2 (Rank 6)
└── GPU 3 (Rank 3)                   └── GPU 3 (Rank 7)
     ↑
     Main Process IP: 192.168.1.100
     Port: 29500

Communication:
- All processes connect to main process (Node 0)
- NCCL backend for GPU-to-GPU communication
- Gloo/MPI for CPU communication
```

**Key Concepts:**
- **num_machines:** Total number of nodes (e.g., 2)
- **machine_rank:** Current node's rank (0 for main, 1 for worker)
- **num_processes:** Total GPUs across all nodes (8 for 2 nodes × 4 GPUs)
- **main_process_ip:** IP address of Node 0
- **main_process_port:** Port for communication (default: 29500)

---

## Method 1: Configuration File (Recommended)

### Step 1: Create Config on Each Node

**Node 0 (Main Process):**
```yaml
# config_node0.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
machine_rank: 0                    # Main process
num_machines: 2                     # Total nodes
num_processes: 8                    # Total GPUs (2 nodes × 4 GPUs)
main_process_ip: 192.168.1.100     # This node's IP
main_process_port: 29500
rdzv_backend: c10d                  # Recommended for multi-node
mixed_precision: fp16
same_network: true                  # All nodes on same LAN
```

**Node 1 (Worker):**
```yaml
# config_node1.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
machine_rank: 1                    # Worker node
num_machines: 2
num_processes: 8
main_process_ip: 192.168.1.100     # Main process IP (Node 0)
main_process_port: 29500
rdzv_backend: c10d
mixed_precision: fp16
same_network: true
```

### Step 2: Launch on Each Node

**On Node 0:**
```bash
accelerate launch --config_file config_node0.yaml train.py
```

**On Node 1 (simultaneously):**
```bash
accelerate launch --config_file config_node1.yaml train.py
```

---

## Method 2: Command-Line Arguments

**On Node 0:**
```bash
accelerate launch \
    --num_processes 8 \
    --num_machines 2 \
    --machine_rank 0 \
    --main_process_ip 192.168.1.100 \
    --main_process_port 29500 \
    --rdzv_backend c10d \
    --mixed_precision fp16 \
    train.py
```

**On Node 1:**
```bash
accelerate launch \
    --num_processes 8 \
    --num_machines 2 \
    --machine_rank 1 \
    --main_process_ip 192.168.1.100 \
    --main_process_port 29500 \
    --rdzv_backend c10d \
    --mixed_precision fp16 \
    train.py
```

---

## Method 3: SLURM (Cluster)

### SLURM Submission Script

**File:** `submit_multinode.sh`

```bash
#!/bin/bash
#SBATCH --job-name=accelerate_train
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --gres=gpu:4                 # 4 GPUs per node
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env

# Get main process IP (first node)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export GPUS_PER_NODE=4

# Launch with accelerate
srun accelerate launch \
    --num_processes $(($SLURM_NNODES * $GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_backend c10d \
    --mixed_precision fp16 \
    train.py \
    --batch_size 32 \
    --epochs 100
```

**Submit:**
```bash
sbatch submit_multinode.sh
```

---

## Training Script (train.py)

```python
from accelerate import Accelerator

def main():
    # Initialize Accelerator (reads config automatically)
    accelerator = Accelerator()

    # Print info on main process only
    if accelerator.is_main_process:
        print(f"Training on {accelerator.num_processes} GPUs across {accelerator.state.num_machines} nodes")
        print(f"Current node: {accelerator.state.machine_rank}")

    # Load model and data
    model = MyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dataloader = DataLoader(dataset, batch_size=32)

    # Prepare for distributed training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Training loop (identical to single-node)
    for epoch in range(100):
        model.train()
        for batch in dataloader:
            outputs = model(batch)
            loss = loss_fn(outputs, labels)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        # Synchronize before evaluation
        accelerator.wait_for_everyone()

        # Evaluate on main process
        if accelerator.is_main_process:
            eval_loss = evaluate(model)
            print(f"Epoch {epoch}: Loss = {eval_loss:.4f}")

if __name__ == "__main__":
    main()
```

---

## Troubleshooting

### 1. Connection Timeout
```
RuntimeError: Connection timed out
```

**Causes:**
- Firewall blocking port 29500
- Wrong main_process_ip
- Nodes not on same network

**Solutions:**
```bash
# Check connectivity
ping 192.168.1.100  # From worker node

# Check port open
nc -zv 192.168.1.100 29500

# Open firewall port
sudo firewall-cmd --add-port=29500/tcp --permanent
sudo firewall-cmd --reload
```

---

### 2. Rank Mismatch
```
RuntimeError: Expected world_size=8, got 4
```

**Cause:** num_processes doesn't match actual GPUs

**Solution:**
```yaml
# Correct calculation
num_machines: 2
GPUs per node: 4
num_processes: 8  # Must equal num_machines × GPUs per node
```

---

### 3. Same Network Issues
```
Warning: Nodes may not be on same network
```

**Solution:**
```yaml
same_network: false  # Set to false for cloud/cross-datacenter
```

---

### 4. NCCL Initialization Failure
```
RuntimeError: NCCL error in: torch.distributed
```

**Solutions:**
```bash
# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"

# Set NCCL debug mode
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Force NCCL socket interface (if multiple network interfaces)
export NCCL_SOCKET_IFNAME=eth0  # Or your interface name
```

---

## Advanced: Manual Process Spawning

**For fine-grained control:**

```python
import torch.distributed as dist
from accelerate import Accelerator

def main_worker(rank, world_size, master_addr, master_port):
    # Manually initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )

    # Now use Accelerator
    accelerator = Accelerator()
    model, optimizer, dataloader = accelerator.prepare(...)

    # Training loop
    for batch in dataloader:
        train_step(batch)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--master_addr", type=str, required=True)
    parser.add_argument("--master_port", type=int, default=29500)
    args = parser.parse_args()

    main_worker(args.rank, args.world_size, args.master_addr, args.master_port)
```

---

## Performance Optimization

### 1. Use Fastest Interconnect
- **InfiniBand (IB):** Best (100+ Gbps)
- **RoCE (RDMA over Converged Ethernet):** Good (40-100 Gbps)
- **10G Ethernet:** Acceptable (10 Gbps)
- **1G Ethernet:** Poor (will bottleneck)

```bash
# Check network interface
ibstat  # For InfiniBand
ifconfig  # For Ethernet
```

### 2. Tune NCCL Parameters
```bash
# Increase NCCL IB timeout (for slow networks)
export NCCL_IB_TIMEOUT=22

# Enable NCCL tree algorithm (better for many nodes)
export NCCL_ALGO=Tree

# Disable NCCL P2P (if causing issues)
export NCCL_P2P_DISABLE=1
```

### 3. Gradient Accumulation for Small Batches
```python
# Instead of small per-GPU batch size across many nodes
accelerator = Accelerator(
    gradient_accumulation_steps=4  # Effective batch size × 4
)
```

---

## Verification Script

**test_multinode.py:**
```python
from accelerate import Accelerator
import torch

accelerator = Accelerator()

# Print process information
print(f"Rank {accelerator.process_index}/{accelerator.num_processes} "
      f"on node {accelerator.state.machine_rank}/{accelerator.state.num_machines}")

# Test all-reduce
tensor = torch.tensor([accelerator.process_index], dtype=torch.float32, device=accelerator.device)
accelerator.print(f"Before all-reduce: {tensor.item()}")

# Sum across all processes
torch.distributed.all_reduce(tensor)
accelerator.print(f"After all-reduce: {tensor.item()}")

expected_sum = sum(range(accelerator.num_processes))
assert tensor.item() == expected_sum, f"All-reduce failed! Got {tensor.item()}, expected {expected_sum}"

accelerator.print("✅ Multi-node setup successful!")
```

**Run:**
```bash
# On Node 0
accelerate launch --config_file config_node0.yaml test_multinode.py

# On Node 1
accelerate launch --config_file config_node1.yaml test_multinode.py
```

**Expected Output:**
```
Rank 0/8 on node 0/2
Rank 1/8 on node 0/2
...
Rank 4/8 on node 1/2
...
Before all-reduce: 0.0
After all-reduce: 28.0  # Sum of 0+1+2+3+4+5+6+7
✅ Multi-node setup successful!
```

---

## Best Practices

1. **Synchronize Codebases:** Use shared filesystem or rsync
2. **Test Single-Node First:** Verify training works on one node
3. **Start Small:** Begin with 2 nodes, then scale
4. **Monitor Network:** Use `iftop` or `nload` to check bandwidth usage
5. **Use Same Environment:** Identical PyTorch, CUDA, NCCL versions
6. **Checkpoint Frequently:** Multi-node training more prone to failures

---

## Related Files
- **Multi-node config template:** `examples/config_yaml_templates/multi_node.yaml`
- **SLURM submission script:** `examples/slurm/submit_multinode.sh`
- **Launch utilities:** `src/accelerate/utils/launch.py`
- **AcceleratorState:** `src/accelerate/state.py:PartialState`
