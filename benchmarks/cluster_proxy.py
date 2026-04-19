#!/usr/bin/env python3
"""
Multi-GPU FSDP proxy benchmark for Phi-reduced vs baseline models.

This benchmark is intentionally lightweight and can run in three modes:
- CPU/single GPU fallback: compares parameter count, latency and memory footprint.
- Multi-GPU FSDP mode: uses torch.distributed and FSDP when >=2 CUDA devices exist.

Refs: Paper §18.5.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.multiprocessing import SimpleQueue
from torch.multiprocessing.spawn import spawn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


@dataclass
class RunMetrics:
    mode: str
    world_size: int
    device: str
    baseline_params: int
    phi_params: int
    baseline_step_ms: float
    phi_step_ms: float
    baseline_peak_mem_mb: float
    phi_peak_mem_mb: float
    profile_name: str
    degree: int
    max_dimension: int
    precision: str
    observable_family: str
    rank_budget: str

    def summary(self) -> str:
        speedup = self.baseline_step_ms / max(self.phi_step_ms, 1e-9)
        mem_delta = self.baseline_peak_mem_mb - self.phi_peak_mem_mb
        return (
            f"mode={self.mode}, world_size={self.world_size}, device={self.device}, "
            f"params(baseline/phi)={self.baseline_params}/{self.phi_params}, "
            f"step_ms(baseline/phi)={self.baseline_step_ms:.3f}/{self.phi_step_ms:.3f}, "
            f"speedup={speedup:.3f}x, mem_delta_mb={mem_delta:.2f}, "
            f"profile={self.profile_name}, precision={self.precision}, rank={self.rank_budget}"
        )


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _rank_scale(rank_budget: str) -> float:
    if rank_budget == "low":
        return 0.75
    if rank_budget == "high":
        return 1.25
    return 1.0


def _precision_dtype(precision: str) -> torch.dtype:
    if precision == "fp64":
        return torch.float64
    return torch.float32


def _make_models(
    device: torch.device,
    degree: int,
    max_dimension: int,
    precision: str,
    rank_budget: str,
) -> Dict[str, torch.nn.Module]:
    dtype = _precision_dtype(precision)
    scale = _rank_scale(rank_budget)
    phi_hidden = int(max(512, min(4096, max_dimension * 24 * scale)))
    baseline_hidden = int(max(1024, min(4096, max(phi_hidden * 1.8, degree * 48))))

    baseline = torch.nn.Sequential(
        torch.nn.Linear(512, baseline_hidden),
        torch.nn.GELU(),
        torch.nn.Linear(baseline_hidden, 512),
    ).to(device=device, dtype=dtype)
    phi = torch.nn.Sequential(
        torch.nn.Linear(512, phi_hidden),
        torch.nn.GELU(),
        torch.nn.Linear(phi_hidden, 512),
    ).to(device=device, dtype=dtype)
    return {"baseline": baseline, "phi": phi}


def _train_steps(model: torch.nn.Module, device: torch.device, steps: int = 20, precision: str = "fp32") -> tuple[float, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dtype = _precision_dtype(precision)
    batch = torch.randn(64, 512, device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(steps):
        out = model(batch)
        loss = (out.square().mean() + out.mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    dt_ms = (time.perf_counter() - t0) * 1000.0 / steps

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    else:
        peak_mb = 0.0
    return dt_ms, peak_mb


def _single_process_run(
    device: torch.device,
    degree: int,
    max_dimension: int,
    precision: str,
    observable_family: str,
    rank_budget: str,
    profile_name: str,
) -> RunMetrics:
    models = _make_models(device, degree, max_dimension, precision, rank_budget)
    baseline_step_ms, baseline_peak_mb = _train_steps(models["baseline"], device, precision=precision)
    phi_step_ms, phi_peak_mb = _train_steps(models["phi"], device, precision=precision)
    return RunMetrics(
        mode="single-process",
        world_size=1,
        device=str(device),
        baseline_params=_count_params(models["baseline"]),
        phi_params=_count_params(models["phi"]),
        baseline_step_ms=baseline_step_ms,
        phi_step_ms=phi_step_ms,
        baseline_peak_mem_mb=baseline_peak_mb,
        phi_peak_mem_mb=phi_peak_mb,
        profile_name=profile_name,
        degree=degree,
        max_dimension=max_dimension,
        precision=precision,
        observable_family=observable_family,
        rank_budget=rank_budget,
    )


def _setup_dist(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29507")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _fsdp_worker(
    rank: int,
    world_size: int,
    queue: SimpleQueue,
    steps: int,
    degree: int,
    max_dimension: int,
    precision: str,
    observable_family: str,
    rank_budget: str,
    profile_name: str,
) -> None:
    _setup_dist(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    models = _make_models(device, degree, max_dimension, precision, rank_budget)
    wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=512 * 512)
    baseline = FSDP(models["baseline"], auto_wrap_policy=wrap_policy, device_id=rank)
    phi = FSDP(models["phi"], auto_wrap_policy=wrap_policy, device_id=rank)

    baseline_step_ms, baseline_peak_mb = _train_steps(baseline, device, steps=steps, precision=precision)
    phi_step_ms, phi_peak_mb = _train_steps(phi, device, steps=steps, precision=precision)

    if rank == 0:
        queue.put(
            RunMetrics(
                mode="fsdp",
                world_size=world_size,
                device=f"cuda:0..{world_size-1}",
                baseline_params=_count_params(models["baseline"]),
                phi_params=_count_params(models["phi"]),
                baseline_step_ms=baseline_step_ms,
                phi_step_ms=phi_step_ms,
                baseline_peak_mem_mb=baseline_peak_mb,
                phi_peak_mem_mb=phi_peak_mb,
                profile_name=profile_name,
                degree=degree,
                max_dimension=max_dimension,
                precision=precision,
                observable_family=observable_family,
                rank_budget=rank_budget,
            )
        )
    dist.destroy_process_group()


def run_proxy_benchmark(
    steps: int = 20,
    degree: int = 32,
    max_dimension: int = 48,
    precision: str = "fp32",
    observable_family: str = "hybrid-polynomial-trig",
    rank_budget: str = "medium",
    profile_name: str = "default",
) -> RunMetrics:
    cuda_devices = torch.cuda.device_count()
    if cuda_devices < 2:
        device = torch.device("cuda:0" if cuda_devices == 1 else "cpu")
        return _single_process_run(
            device,
            degree=degree,
            max_dimension=max_dimension,
            precision=precision,
            observable_family=observable_family,
            rank_budget=rank_budget,
            profile_name=profile_name,
        )

    queue: SimpleQueue = SimpleQueue()
    spawn(
        _fsdp_worker,
        args=(
            cuda_devices,
            queue,
            steps,
            degree,
            max_dimension,
            precision,
            observable_family,
            rank_budget,
            profile_name,
        ),
        nprocs=cuda_devices,
        join=True,
    )
    metrics = queue.get()
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cluster proxy benchmark")
    parser.add_argument("--steps", type=int, default=20, help="Training steps per model")
    parser.add_argument("--profile-name", default="default", help="Profile identifier")
    parser.add_argument("--degree", type=int, default=32, help="Compile profile degree")
    parser.add_argument("--max-dimension", type=int, default=48, help="Compile profile max dimension")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp64"], help="Numeric precision")
    parser.add_argument("--observable-family", default="hybrid-polynomial-trig", help="Observable family label")
    parser.add_argument("--rank-budget", default="medium", choices=["low", "medium", "high"], help="Rank budget label")
    parser.add_argument(
        "--output",
        default="artifacts/cluster_proxy_metrics.json",
        help="Output JSON path for benchmark metrics",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metrics = run_proxy_benchmark(
        steps=args.steps,
        degree=args.degree,
        max_dimension=args.max_dimension,
        precision=args.precision,
        observable_family=args.observable_family,
        rank_budget=args.rank_budget,
        profile_name=args.profile_name,
    )
    print(metrics.summary())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    print(f"Saved metrics to {output_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

