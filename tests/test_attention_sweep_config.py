"""Check experiment coverage without launching GPU jobs."""
from argparse import Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_attention import IMPLEMENTATIONS, build_cases, tuning_implementation


def test_full_sweep_includes_compiled_and_matched_square_tiles():
    args = Namespace(seq_lengths=[2**i for i in range(7, 17)], dims=[16, 32, 64, 128],
                     dtypes=['bfloat16', 'float32'], implementations=IMPLEMENTATIONS,
                     tile_sensitivity=True, tile_seq_lengths=[1024, 4096, 16384, 65536],
                     tile_sizes=[32, 64, 128], tile_pytorch_sizes=[256, 512, 1024],
                     tile_implementations=['flash_pytorch', 'flash_pytorch_compiled', 'flash_triton'])
    cases = build_cases(args)
    assert len(cases) == len(set(cases)) == 440
    main = [c for c in cases if c[-1] is None]
    assert len(main) == 320
    assert len([c for c in main if c[3] == 'flash_pytorch_compiled']) == 80
    eager = {(n, d, dtype, tile) for n, d, dtype, impl, tile in cases
             if impl == 'flash_pytorch' and tile is not None}
    compiled = {(n, d, dtype, tile) for n, d, dtype, impl, tile in cases
                if impl == 'flash_pytorch_compiled' and tile is not None}
    assert eager == compiled and len(eager) == 24
    assert all(tile[0] == tile[1] for *_, tile in compiled)
    assert tuning_implementation('flash_pytorch_compiled') == 'flash_pytorch'
