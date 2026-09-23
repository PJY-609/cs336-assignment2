# Triton tile-size sensitivity on a single H200

Batch size 1; causal masking; median CUDA-event latency in milliseconds.
Naive = existing compiled dense PyTorch; Flash PyTorch = eager tiled Python loops.
Tile tuning, correctness checks, input generation and compilation are outside timing.
Peak memory is total PyTorch-allocated GPU memory in decimal GB, including preallocated inputs.
Forward and forward+backward memory are measured in separate untimed passes.
Partial results are saved after each case. Blank cells are not measured values.

| Implementation | N | D | Dtype | Q tile | K tile | Forward ms | Backward ms | Fwd+bwd ms | Forward peak GB | Fwd+bwd peak GB | Status |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| flash_triton | 1024 | 64 | bfloat16 | 32 | 32 | 0.0277 | 0.1004 | 0.1702 | 0.0678 | 0.0686 | ok |
| flash_triton | 1024 | 64 | bfloat16 | 32 | 64 | 0.0199 | 0.1053 | 0.1807 | 0.0678 | 0.0686 | ok |
| flash_triton | 1024 | 64 | bfloat16 | 32 | 128 | 0.0185 | 0.1092 | 0.1591 | 0.0678 | 0.0686 | ok |
| flash_triton | 1024 | 64 | bfloat16 | 64 | 32 | 0.0217 | 0.0985 | 0.1701 | 0.0678 | 0.0686 | ok |
| flash_triton | 1024 | 64 | bfloat16 | 64 | 64 | 0.0182 | 0.1032 | 0.1738 | 0.0678 | 0.0686 | ok |
| flash_triton | 1024 | 64 | bfloat16 | 64 | 128 | 0.0188 | 0.1078 | 0.1757 | 0.0678 | 0.0686 | ok |
| flash_triton | 1024 | 64 | bfloat16 | 128 | 32 | 0.0515 | 0.1029 | 0.1703 | 0.0678 | 0.0686 | ok |
| flash_triton | 1024 | 64 | bfloat16 | 128 | 64 | 0.0441 | 0.1132 | 0.1991 | 0.0678 | 0.0686 | ok |
| flash_triton | 1024 | 64 | bfloat16 | 128 | 128 | 0.0533 | 0.1643 | 0.2125 | 0.0678 | 0.0686 | ok |
| flash_triton | 1024 | 64 | float32 | 32 | 32 | 0.1094 | 0.4651 | 0.5723 | 0.0684 | 0.0692 | ok |
| flash_triton | 1024 | 64 | float32 | 32 | 64 | 0.1126 | 0.7186 | 0.8283 | 0.0684 | 0.0692 | ok |
| flash_triton | 1024 | 64 | float32 | 32 | 128 | 0.7591 | 8.6258 | 9.3818 | 0.0684 | 0.0692 | ok |
| flash_triton | 1024 | 64 | float32 | 64 | 32 | 0.2502 | 0.7215 | 0.9687 | 0.0684 | 0.0692 | ok |
| flash_triton | 1024 | 64 | float32 | 64 | 64 | 2.6323 | 6.8438 | 9.4665 | 0.0684 | 0.0692 | ok |
| flash_triton | 1024 | 64 | float32 | 64 | 128 | 2.5322 | 15.1960 | 17.7268 | 0.0684 | 0.0692 | ok |
| flash_triton | 1024 | 64 | float32 | 128 | 32 | 0.7405 | 2.9175 | 3.6623 | 0.0684 | 0.0692 | ok |
| flash_triton | 1024 | 64 | float32 | 128 | 64 | 8.7422 | 11.9273 | 20.6540 | 0.0684 | 0.0692 | ok |
| flash_triton | 1024 | 64 | float32 | 128 | 128 |  |  |  |  |  | resource_limit |
| flash_triton | 4096 | 64 | bfloat16 | 32 | 32 | 0.0928 | 0.1614 | 0.2443 | 0.0697 | 0.0729 | ok |
| flash_triton | 4096 | 64 | bfloat16 | 32 | 64 | 0.0618 | 0.2066 | 0.2655 | 0.0697 | 0.0729 | ok |
| flash_triton | 4096 | 64 | bfloat16 | 32 | 128 | 0.0508 | 0.3604 | 0.4068 | 0.0697 | 0.0729 | ok |
| flash_triton | 4096 | 64 | bfloat16 | 64 | 32 | 0.0690 | 0.1417 | 0.2051 | 0.0697 | 0.0729 | ok |
| flash_triton | 4096 | 64 | bfloat16 | 64 | 64 | 0.0551 | 0.1797 | 0.2301 | 0.0697 | 0.0729 | ok |
| flash_triton | 4096 | 64 | bfloat16 | 64 | 128 | 0.0570 | 0.3505 | 0.4002 | 0.0697 | 0.0729 | ok |
| flash_triton | 4096 | 64 | bfloat16 | 128 | 32 | 0.1868 | 0.2578 | 0.4420 | 0.0697 | 0.0729 | ok |
| flash_triton | 4096 | 64 | bfloat16 | 128 | 64 | 0.1570 | 0.2603 | 0.4116 | 0.0697 | 0.0729 | ok |
| flash_triton | 4096 | 64 | bfloat16 | 128 | 128 | 0.2041 | 0.5859 | 0.7848 | 0.0697 | 0.0729 | ok |
| flash_triton | 4096 | 64 | float32 | 32 | 32 | 0.4200 | 1.8027 | 2.2196 | 0.0724 | 0.0755 | ok |
| flash_triton | 4096 | 64 | float32 | 32 | 64 | 0.4307 | 2.8516 | 3.2725 | 0.0724 | 0.0755 | ok |
| flash_triton | 4096 | 64 | float32 | 32 | 128 | 3.6193 | 35.8498 | 39.5232 | 0.0724 | 0.0755 | ok |
| flash_triton | 4096 | 64 | float32 | 64 | 32 | 1.0020 | 2.8306 | 3.8242 | 0.0724 | 0.0755 | ok |
| flash_triton | 4096 | 64 | float32 | 64 | 64 | 10.8426 | 30.8699 | 41.7247 | 0.0724 | 0.0755 | ok |
| flash_triton | 4096 | 64 | float32 | 64 | 128 | 11.9159 | 73.5413 | 85.3799 | 0.0724 | 0.0755 | ok |
| flash_triton | 4096 | 64 | float32 | 128 | 32 | 3.0433 | 14.8827 | 17.7974 | 0.0724 | 0.0755 | ok |
| flash_triton | 4096 | 64 | float32 | 128 | 64 | 36.9103 | 55.1978 | 92.0316 | 0.0724 | 0.0755 | ok |
| flash_triton | 4096 | 64 | float32 | 128 | 128 |  |  |  |  |  | resource_limit |
| flash_triton | 16384 | 64 | bfloat16 | 32 | 32 | 0.7018 | 1.4854 | 2.1851 | 0.0777 | 0.0902 | ok |
| flash_triton | 16384 | 64 | bfloat16 | 32 | 64 | 0.5207 | 1.1858 | 1.7021 | 0.0777 | 0.0902 | ok |
| flash_triton | 16384 | 64 | bfloat16 | 32 | 128 | 0.4989 | 1.6304 | 2.1278 | 0.0777 | 0.0902 | ok |
| flash_triton | 16384 | 64 | bfloat16 | 64 | 32 | 0.3347 | 0.9679 | 1.2980 | 0.0777 | 0.0902 | ok |
| flash_triton | 16384 | 64 | bfloat16 | 64 | 64 | 0.3036 | 0.7630 | 1.0650 | 0.0777 | 0.0902 | ok |
| flash_triton | 16384 | 64 | bfloat16 | 64 | 128 | 0.3001 | 1.3724 | 1.6688 | 0.0777 | 0.0902 | ok |
| flash_triton | 16384 | 64 | bfloat16 | 128 | 32 | 0.7309 | 1.2428 | 1.9719 | 0.0777 | 0.0902 | ok |
| flash_triton | 16384 | 64 | bfloat16 | 128 | 64 | 0.6180 | 1.2183 | 1.8311 | 0.0777 | 0.0902 | ok |
| flash_triton | 16384 | 64 | bfloat16 | 128 | 128 | 0.7636 | 2.2488 | 2.9677 | 0.0777 | 0.0902 | ok |
| flash_triton | 16384 | 64 | float32 | 32 | 32 | 6.0336 | 24.0778 | 30.0458 | 0.0881 | 0.1008 | ok |
| flash_triton | 16384 | 64 | float32 | 32 | 64 | 5.7529 | 23.7872 | 29.4644 | 0.0881 | 0.1008 | ok |
| flash_triton | 16384 | 64 | float32 | 32 | 128 | 55.4348 | 233.4512 | 288.8642 | 0.0881 | 0.1008 | ok |
| flash_triton | 16384 | 64 | float32 | 64 | 32 | 6.7233 | 26.8455 | 33.5797 | 0.0881 | 0.1008 | ok |
| flash_triton | 16384 | 64 | float32 | 64 | 64 | 75.9321 | 221.0599 | 297.4666 | 0.0881 | 0.1008 | ok |
| flash_triton | 16384 | 64 | float32 | 64 | 128 | 132.0331 | 479.8576 | 612.3162 | 0.0881 | 0.1008 | ok |
| flash_triton | 16384 | 64 | float32 | 128 | 32 | 12.1111 | 177.1462 | 189.1607 | 0.0881 | 0.1008 | ok |
| flash_triton | 16384 | 64 | float32 | 128 | 64 | 192.0925 | 382.9095 | 573.9219 | 0.0881 | 0.1008 | ok |
| flash_triton | 16384 | 64 | float32 | 128 | 128 |  |  |  |  |  | resource_limit |
| flash_triton | 65536 | 64 | bfloat16 | 32 | 32 | 11.0698 | 22.6790 | 33.7564 | 0.1093 | 0.1596 | ok |
| flash_triton | 65536 | 64 | bfloat16 | 32 | 64 | 8.0140 | 18.1845 | 26.2205 | 0.1093 | 0.1596 | ok |
| flash_triton | 65536 | 64 | bfloat16 | 32 | 128 | 7.8999 | 18.3821 | 26.2588 | 0.1093 | 0.1596 | ok |
| flash_triton | 65536 | 64 | bfloat16 | 64 | 32 | 4.5344 | 13.2930 | 17.8677 | 0.1093 | 0.1596 | ok |
| flash_triton | 65536 | 64 | bfloat16 | 64 | 64 | 4.2149 | 11.2343 | 15.3920 | 0.1093 | 0.1596 | ok |
| flash_triton | 65536 | 64 | bfloat16 | 64 | 128 | 4.5622 | 14.3650 | 18.8749 | 0.1093 | 0.1596 | ok |
| flash_triton | 65536 | 64 | bfloat16 | 128 | 32 | 6.2020 | 15.2957 | 21.3389 | 0.1093 | 0.1596 | ok |
| flash_triton | 65536 | 64 | bfloat16 | 128 | 64 | 5.6505 | 14.7040 | 20.4578 | 0.1093 | 0.1596 | ok |
| flash_triton | 65536 | 64 | bfloat16 | 128 | 128 | 6.8484 | 34.9950 | 41.8348 | 0.1093 | 0.1596 | ok |
| flash_triton | 65536 | 64 | float32 | 32 | 32 | 96.0803 | 383.3432 | 479.2498 | 0.1513 | 0.2019 | ok |
| flash_triton | 65536 | 64 | float32 | 32 | 64 | 92.1043 | 375.6523 | 467.1992 | 0.1513 | 0.2019 | ok |
| flash_triton | 65536 | 64 | float32 | 32 | 128 | 925.3262 | 2477.1211 | 3435.5862 | 0.1513 | 0.2019 | ok |
| flash_triton | 65536 | 64 | float32 | 64 | 32 | 107.0612 | 423.5500 | 530.6665 | 0.1513 | 0.2019 | ok |
| flash_triton | 65536 | 64 | float32 | 64 | 64 | 1214.1405 | 3510.8921 | 4730.6333 | 0.1513 | 0.2019 | ok |
| flash_triton | 65536 | 64 | float32 | 64 | 128 | 2108.6143 | 7676.4131 | 9789.8750 | 0.1513 | 0.2019 | ok |
| flash_triton | 65536 | 64 | float32 | 128 | 32 | 130.1172 | 2763.8479 | 2892.5952 | 0.1513 | 0.2019 | ok |
| flash_triton | 65536 | 64 | float32 | 128 | 64 | 2925.3042 | 6044.6201 | 8968.3291 | 0.1513 | 0.2019 | ok |
| flash_triton | 65536 | 64 | float32 | 128 | 128 |  |  |  |  |  | resource_limit |
