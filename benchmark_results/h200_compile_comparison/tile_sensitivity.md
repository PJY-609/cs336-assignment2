# Tiled attention tile-size sensitivity on a single H200

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
