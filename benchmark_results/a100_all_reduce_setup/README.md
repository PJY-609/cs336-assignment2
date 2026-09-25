# All-reduce setup validation

Prepared against repository commit `81bb427` on 2026-09-25 UTC.
These are setup smoke tests, **not the full experiment or reportable results**.

- Locked environment: Python 3.12, PyTorch 2.11.0+cu130, CUDA 13.0,
  NCCL 2.28.9; NVIDIA driver 580.126.16.
- Six visible NVIDIA A100-SXM4-80GB GPUs; topology reports NV12 for
  every GPU pair. Provider confirmation of the physical NVSwitch topology
  remains a separate checklist item.
- Gloo/CPU: 2 ranks, 4,096 and 40,000 bytes per rank; both passed.
- NCCL/CUDA: 2, 4, and 6 ranks, 1,000,000 bytes per rank; all passed.
- Each case used 5 warmups and 3 measured iterations, with SUM validation
  before warmup and after measurement.

Each subdirectory preserves environment/topology metadata, runner source,
lockfile, launch records, raw rank timings, and a summary CSV. The default
full-run directory remains unused. See `DISTRIBUTED_COMMUNICATION_EXPERIMENT.md`
for setup and launch instructions. Use new output paths when repeating smoke
checks, as the runner refuses to overwrite existing output directories.
