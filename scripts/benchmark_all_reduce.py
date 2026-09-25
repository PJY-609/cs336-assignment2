"""Single-node FP32 SUM all-reduce: one bounded torchrun launch per case."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, UTC
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def capture(command):
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def summarize(samples):
    maxima = [max(values) for values in zip(*samples, strict=True)]
    quartiles = statistics.quantiles(maxima, n=4, method="inclusive") if len(maxima) > 1 else maxima * 3
    return {
        "iteration_max_ms": maxima,
        "median_ms": statistics.median(maxima),
        "q1_ms": quartiles[0], "q3_ms": quartiles[2],
        "iqr_ms": quartiles[2] - quartiles[0],
        "rank_medians_ms": [statistics.median(rank) for rank in samples],
    }


def worker(args):
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(args.backend, timeout=timedelta(seconds=args.timeout_seconds))
    try:
        tensor = torch.empty(args.tensor_bytes // 4, dtype=torch.float32, device=device)

        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def prepare():
            tensor.fill_(rank + 1)
            dist.barrier()
            synchronize()

        prepare()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
        synchronize()
        expected = world_size * (world_size + 1) // 2
        if not bool(torch.all(tensor == expected).item()):
            raise RuntimeError(f"Rank {rank}: SUM validation failed; expected {expected}")
        samples = []
        for iteration in range(args.warmup + args.iterations):
            prepare()
            start = time.perf_counter_ns()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
            synchronize()
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            if iteration >= args.warmup:
                samples.append(elapsed_ms)
        # Keep validation and object collection outside every timed interval.
        if not bool(torch.all(tensor == expected).item()):
            raise RuntimeError(f"Rank {rank}: final SUM validation failed")
        gathered = [None] * world_size
        dist.all_gather_object(gathered, samples)
        if rank == 0:
            save_json(args.output, {
                "status": "ok", "backend": args.backend, "device": args.device,
                "dtype": "float32", "tensor_bytes": args.tensor_bytes,
                "elements": args.tensor_bytes // 4, "world_size": world_size,
                "repeat": args.repeat, "warmup": args.warmup, "iterations": args.iterations,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "metric": "synchronized host-observed call latency", "unit": "ms",
                "validated": True, "rank_samples_ms": gathered, **summarize(gathered),
            })
    finally:
        dist.destroy_process_group()


def metadata(args):
    import torch

    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "hostname": platform.node(), "platform": platform.platform(),
        "python": sys.version, "executable": sys.executable,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version() if torch.cuda.is_available() else None,
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {key: value for key, value in os.environ.items()
                        if key.startswith(("NCCL_", "TORCH_NCCL_")) or key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")},
        "gpus": [{"logical_id": i, "name": torch.cuda.get_device_name(i),
                  "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                 for i in range(torch.cuda.device_count())],
        "git_commit": capture(["git", "rev-parse", "HEAD"]),
        "git_status": capture(["git", "status", "--short"]),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "nvidia_smi": capture(["nvidia-smi"]),
        "topology": capture(["nvidia-smi", "topo", "-m"]),
        "gpu_inventory": capture(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id", "--format=csv"]),
    }


def run_matrix(args):
    import torch

    gpu_ids = args.gpu_ids.split(",") if args.gpu_ids else list(range(torch.cuda.device_count()))
    if args.device == "cuda" and (not torch.cuda.is_available() or max(args.world_sizes) > len(gpu_ids)):
        raise ValueError("Not enough selected CUDA devices for the requested world sizes")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must be unique")
    # Refuse reuse so a smoke test or rerun cannot silently overwrite measurements.
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "environment.json", metadata(args))
    (args.output / "benchmark_all_reduce.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "uv.lock").write_bytes((ROOT / "uv.lock").read_bytes())
    rows = []
    for repeat in range(1, args.repeats + 1):
        for world_size in args.world_sizes:
            for size in args.sizes:
                name = f"repeat{repeat}_ranks{world_size}_bytes{size}"
                output = (args.output / f"{name}.json").resolve()
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                           f"--nproc-per-node={world_size}", "--max-restarts=0", str(Path(__file__).resolve()),
                           "--worker", "--backend", args.backend, "--device", args.device,
                           "--tensor-bytes", str(size), "--warmup", str(args.warmup),
                           "--iterations", str(args.iterations), "--repeat", str(repeat),
                           "--timeout-seconds", str(args.timeout_seconds), "--output", str(output)]
                env = os.environ.copy()
                env.setdefault("OMP_NUM_THREADS", "1")
                selected = ",".join(map(str, gpu_ids[:world_size])) if args.device == "cuda" else ""
                env["CUDA_VISIBLE_DEVICES"] = selected
                record = {"tensor_bytes": size, "world_size": world_size, "repeat": repeat,
                          "warmup": args.warmup, "iterations": args.iterations, "selected_gpu_ids": selected,
                          "status": "error", "command": command}
                print(name, flush=True)
                start = time.monotonic()
                with (args.output / f"{name}.log").open("w") as log:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = process.wait(timeout=args.timeout_seconds)
                    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                        # Kill remaining workers even if torchrun itself already exited.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                        record["status"] = "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "interrupted"
                        code = process.returncode
                    record["returncode"] = code
                record["wall_seconds"] = time.monotonic() - start
                if code == 0 and output.exists() and record["status"] == "error":
                    result = json.loads(output.read_text())
                    record.update({key: result[key] for key in ("status", "median_ms", "q1_ms", "q3_ms", "iqr_ms")})
                save_json(args.output / f"{name}.launch.json", record)
                rows.append({key: value for key, value in record.items() if key != "command"})
                fields = ["tensor_bytes", "world_size", "repeat", "warmup", "iterations", "selected_gpu_ids",
                          "status", "returncode", "wall_seconds", "median_ms", "q1_ms", "q3_ms", "iqr_ms"]
                with (args.output / "summary.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                if record["status"] != "ok":
                    raise RuntimeError(f"{name}: {record['status']}; inspect {args.output / (name + '.log')}")
    print(f"Saved {len(rows)} configurations to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="nccl")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000_000, 10_000_000, 100_000_000, 1_000_000_000], help="Decimal bytes per rank")
    parser.add_argument("--gpu-ids", default=os.environ.get("CUDA_VISIBLE_DEVICES"), help="Ordered physical GPU indices or UUIDs")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=290, help="Wall-clock limit per torchrun launch")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_results/a100_sxm_all_reduce")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tensor-bytes", type=int, default=4096, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.backend, args.device) not in (("gloo", "cpu"), ("nccl", "cuda")):
        parser.error("Use gloo/cpu for debugging or nccl/cuda for measurements")
    if min(args.world_sizes + [args.iterations, args.repeats, args.timeout_seconds]) <= 0 or args.warmup < 5:
        parser.error("Counts must be positive and warmup must be at least 5")
    if any(size <= 0 or size % 4 for size in args.sizes + [args.tensor_bytes]):
        parser.error("Tensor sizes must be positive multiples of four bytes")
    if len(set(args.world_sizes)) != len(args.world_sizes) or len(set(args.sizes)) != len(args.sizes):
        parser.error("World sizes and tensor sizes must be unique")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    worker(arguments) if arguments.worker else run_matrix(arguments)
