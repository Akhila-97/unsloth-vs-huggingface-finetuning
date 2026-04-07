# unsloth-vs-huggingface-finetuning
This project provides a side-by-side performance analysis of fine-tuning the Llama 3.2 3B model. By comparing the standard Hugging Face (TRL/PEFT) workflow against the optimized Unsloth library, this repository demonstrates how custom Triton kernels and efficient memory management can drastically accelerate LLM training.

Fine-tuning a 3B parameter model on a consumer laptop GPU with only 8GB of VRAM is not straightforward. The first attempt using a standard HuggingFace pipeline with batch size 2 and gradient accumulation 4 crashed during the very first backward pass:
torch.AcceleratorError: CUDA error: unknown error
File ".../torch/utils/checkpoint.py", line 325, in backward
    torch.autograd.backward(outputs_with_grad, args_with_grad)
The crash happened inside gradient checkpointing during backpropagation, the model completed the forward pass but ran out of VRAM the moment it tried to compute gradients. To get the baseline running, batch size had to be reduced to 1 and gradient accumulation doubled to 8, keeping the effective batch size at 8 but halving how much the GPU had to hold at once per step.
Unsloth ran at batch size 2 with gradient accumulation 4 without any modification.

Hardware & Setup
 GPU - NVIDIA GeForce RTX 5060 Laptop GPU, VRAM - 7.96 GB, System RAM-32 GB, OS - Linux (WSL2 on Windows), CUDA -12.8, PyTorch - 2.10.0+cu128


Model & Training Configuration
Base model - LLaMA 3.2 3B Instruct, Quantization - 4-bit (NF4), LoRA rank (r) - 16, LoRA alpha - 16, LoRA dropout - 0, Target modules - q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, Trainable parameters - 24,313,856 (0.75% of total), Epochs - 3, Learning rate - 2e-4, Optimizer - adamw_8bit, Precision - bfloat16

Results ----------
**Model Load Time**
- HuggingFace: 197s
- Unsloth: 107s
- Improvement: 1.84x faster

**Total Train Time**
- HuggingFace: 1566s
- Unsloth: 465s
- Improvement: 3.37x faster

**Seconds Per Step**
- HuggingFace: 20.88s
- Unsloth: 6.21s
- Improvement: 3.37x faster

**Peak VRAM**
- HuggingFace: 6.06 GB
- Unsloth: 4.60 GB
- Improvement: 24% less

**Reserved VRAM (buffer)**
- HuggingFace: 17.31 GB
- Unsloth: 4.81 GB
- Improvement: 72% less

**Eval Loss**
- HuggingFace: 1.8183
- Unsloth: 1.9812
- Comparable

**Batch size needed to avoid crash**
- HuggingFace: 1 (crashed at 2)
- Unsloth: 2

Why Unsloth Is Faster -----
Standard HuggingFace runs each operation, matrix multiply, activation function, elementwise operations as a separate GPU kernel. Each kernel reads its inputs from slow HBM (main GPU memory), does its math, and writes its output back to HBM before the next kernel starts. These constant round trips between compute units and memory are the bottleneck.
Unsloth replaces these standard operations with custom Triton kernels that fuse multiple operations together. Intermediate results stay in fast on-chip SRAM and are used immediately for the next operation without ever being written to HBM. This is applied to attention layers (QKV projections), MLP layers (gate/up/down projections), and LoRA operations. You can see this confirmed in the training log:
Unsloth 2026.2.1 patched 28 layers with 28 QKV layers, 28 O layers and 28 MLP layers.
All 28 layers were fully patched because LoRA dropout was set to 0. Unsloth cannot apply its fast kernels to layers with active dropout.

Why VRAM Usage Is So Different --------
The baseline reserved 17.31 GB of VRAM on an 8 GB card, it was spilling heavily into system memory via CUDA unified memory, which is extremely slow. This is because standard gradient checkpointing stores all forward pass activations in VRAM throughout training so they are available during the backward pass.
Unsloth uses a smarter approach: it recomputes activations on the fly during the backward pass instead of storing them upfront. It also smartly offloads gradients when they are not immediately needed:
Unsloth: Will smartly offload gradients to save VRAM!
The result is peak VRAM of 4.60 GB and a reserved buffer of only 4.81 GB, well within the GPU's 7.96 GB limit, with no spill into system memory.

Files
fine_tuning.py -- Unsloth fine-tuning script with VRAM and time, fine_tuning_witoutunsloth.py -- Standard HuggingFace + PEFT baseline with matching metrics
