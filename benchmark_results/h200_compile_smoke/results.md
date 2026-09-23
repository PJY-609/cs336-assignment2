# Attention benchmark on a single H200

Batch size 1; causal masking; median CUDA-event latency in milliseconds.
Naive = existing compiled dense PyTorch; Flash PyTorch = eager tiled Python loops.
flash_pytorch_compiled = the same tiled forward and backward with full-graph torch.compile.
First-call wall times (including compilation when needed) are reported separately in CSV/JSON.
Tile tuning, correctness checks, input generation and compilation are outside timing.
Peak memory is total PyTorch-allocated GPU memory in decimal GB, including preallocated inputs.
Forward and forward+backward memory are measured in separate untimed passes.
Partial results are saved after each case. Blank cells are not measured values.

| Implementation | N | D | Dtype | Q tile | K tile | Forward ms | Backward ms | Fwd+bwd ms | Forward peak GB | Fwd+bwd peak GB | Status |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| naive | 128 | 64 | bfloat16 |  |  | 0.0247 | 0.1526 | 0.2590 | 0.0672 | 0.0673 | ok |
| flash_pytorch | 128 | 64 | bfloat16 | 128 | 128 | 0.2707 | 0.4784 | 0.8708 | 0.0673 | 0.0675 | ok |
| flash_pytorch_compiled | 128 | 64 | bfloat16 | 128 | 128 | 0.0791 | 0.2061 | 0.3801 | 0.0672 | 0.0674 | ok |
| flash_triton | 128 | 64 | bfloat16 | 64 | 32 | 0.0255 | 0.1512 | 0.2565 | 0.0672 | 0.0673 | ok |
| naive | 128 | 64 | float32 |  |  | 0.0548 | 0.1709 | 0.3095 | 0.0674 | 0.0675 | ok |
| flash_pytorch | 128 | 64 | float32 | 128 | 128 | 0.3008 | 0.5120 | 1.3674 | 0.0676 | 0.0679 | ok |
| flash_pytorch_compiled | 128 | 64 | float32 | 128 | 128 | 0.0751 | 0.2211 | 0.3869 | 0.0673 | 0.0676 | ok |
| flash_triton | 128 | 64 | float32 | 16 | 16 | 0.0185 | 0.1298 | 0.2123 | 0.0673 | 0.0674 | ok |
