# Completed H200 benchmark

This run completed in 48 minutes on one NVIDIA H200. All 240 main-sweep cases
succeeded. Of 72 fixed-tile cases, 68 succeeded and four hit the kernel's
shared-memory limit: FP32, dimension 64, query/key tiles 128x128, at each tested
sequence length. No case exhausted GPU device memory.

- [Main results](results.md) and [CSV](results.csv)
- [Tile-sensitivity results](tile_sensitivity.md) and [CSV](tile_sensitivity.csv)
- [Forward peak memory](plots/memory_forward.png)
- [Forward+backward peak memory](plots/memory_forward_backward.png)
- [Forward tile sensitivity](plots/tile_sensitivity_forward.png)
- [Backward tile sensitivity](plots/tile_sensitivity_backward.png)
- [Forward+backward tile sensitivity](plots/tile_sensitivity_forward_backward.png)

PDF versions of the plots are also included. Memory is peak PyTorch-allocated
GPU memory in decimal GB, including resident inputs. Latencies are measured in
milliseconds with `triton.testing.do_bench`.

See [benchmark methodology](../../BENCHMARKING.md) for precision settings,
validation tolerances, timing scope, memory accounting, and tile selection.
`cases/` contains detailed measurements and errors; `tuning/` preserves tile
selection trials. `manifest.json` and `progress.json` record the configuration
and completion status. Source and dependency snapshots preserve the run inputs.
