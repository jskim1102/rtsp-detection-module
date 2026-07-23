#!/usr/bin/env python3
"""GPU worker-pool benchmark (서버/API/DB를 건드리지 않는 synthetic frame 측정).

예:
  .venv/bin/python scripts/benchmark_inference_matrix.py \
    --models yolo26n.pt,yolo26x.pt --counts 1,2,4,8,16 --duration 15

모델 가중치 다운로드와 GPU process 기동이 발생하므로 CTO/user가 라이브 검증 때 실행한다.
JSONL 한 줄이 한 model/camera-count 케이스다.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.inference import FrameRequest, InferenceWorker
from app.inference.models_dir import PRESET_MODELS


@dataclass
class Sample:
    model: str
    cameras: int
    imgsz: int
    batch_max: int
    submitted: int
    received: int
    aggregate_fps: float
    per_camera_fps: float
    p50_latency_ms: float
    p95_latency_ms: float
    gpu_util_avg: float | None
    gpu_util_peak: float | None
    gpu_memory_peak_mb: float | None
    queue_dropped: int


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _gpu_sample() -> tuple[float, float] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        first = completed.stdout.splitlines()[0]
        util, memory = (float(value.strip()) for value in first.split(",")[:2])
        return util, memory
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def _run_case(
    model: str,
    cameras: int,
    *,
    imgsz: int,
    batch_max: int,
    batch_timeout_sec: float,
    warmup_sec: float,
    duration_sec: float,
    source_fps: float,
) -> Sample:
    worker = InferenceWorker(model_name=model)
    worker._BATCH_MAX = batch_max
    worker._BATCH_TIMEOUT_SEC = batch_timeout_sec
    worker.configure_models({model})
    worker.start()
    frame = np.zeros((imgsz * 9 // 16, imgsz, 3), dtype=np.uint8)
    interval = 1.0 / source_fps
    next_submit = {f"cam-{index}": time.perf_counter() for index in range(cameras)}
    latencies: list[float] = []
    gpu_utils: list[float] = []
    gpu_memory: list[float] = []
    received = 0
    started = time.perf_counter()
    measure_at = started + warmup_sec
    finish_at = measure_at + duration_sec
    next_gpu_sample = started

    try:
        while time.perf_counter() < finish_at:
            now = time.perf_counter()
            for source_id, deadline in list(next_submit.items()):
                if now < deadline:
                    continue
                worker.submit(
                    FrameRequest(
                        source_id=source_id,
                        frame=frame,
                        timestamp=time.time(),
                        model_names=[model],
                        imgsz=imgsz,
                    )
                )
                skipped = max(1, math.floor((now - deadline) / interval) + 1)
                next_submit[source_id] = deadline + skipped * interval

            for result in worker.drain_results():
                if now >= measure_at:
                    received += 1
                    latencies.append(max(0.0, (time.time() - result.timestamp) * 1000.0))

            if now >= next_gpu_sample:
                sample = _gpu_sample()
                if sample is not None and now >= measure_at:
                    util, memory = sample
                    gpu_utils.append(util)
                    gpu_memory.append(memory)
                next_gpu_sample = now + 0.5
            time.sleep(0.001)

        # 마지막 completed batch 회수.
        time.sleep(min(0.25, batch_timeout_sec * 4 + 0.02))
        for result in worker.drain_results():
            received += 1
            latencies.append(max(0.0, (time.time() - result.timestamp) * 1000.0))
        stats = worker.get_queue_stats()
    finally:
        worker.stop()

    aggregate_fps = received / duration_sec
    return Sample(
        model=model,
        cameras=cameras,
        imgsz=imgsz,
        batch_max=batch_max,
        submitted=int(stats["submitted"]),
        received=received,
        aggregate_fps=round(aggregate_fps, 3),
        per_camera_fps=round(aggregate_fps / cameras, 3),
        p50_latency_ms=round(statistics.median(latencies), 3) if latencies else 0.0,
        p95_latency_ms=round(_percentile(latencies, 0.95), 3),
        gpu_util_avg=round(statistics.mean(gpu_utils), 2) if gpu_utils else None,
        gpu_util_peak=max(gpu_utils) if gpu_utils else None,
        gpu_memory_peak_mb=max(gpu_memory) if gpu_memory else None,
        queue_dropped=int(stats["dropped"]),
    )


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(PRESET_MODELS))
    parser.add_argument("--counts", default="1,2,4,8,16")
    parser.add_argument("--imgsz", type=int, default=640, choices=(320, 416, 512, 640))
    parser.add_argument("--batch-max", type=int, default=8)
    parser.add_argument("--batch-timeout-ms", type=float, default=8.0)
    parser.add_argument("--warmup", type=float, default=8.0)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--source-fps", type=float, default=30.0)
    args = parser.parse_args()

    models = _csv(args.models)
    invalid = [model for model in models if model not in PRESET_MODELS]
    if invalid:
        parser.error(f"unsupported models: {invalid}")
    counts = [int(value) for value in _csv(args.counts)]
    if any(value < 1 or value > 64 for value in counts):
        parser.error("counts must be between 1 and 64")

    for model in models:
        for cameras in counts:
            sample = _run_case(
                model,
                cameras,
                imgsz=args.imgsz,
                batch_max=max(1, args.batch_max),
                batch_timeout_sec=max(0.0, args.batch_timeout_ms / 1000.0),
                warmup_sec=max(0.0, args.warmup),
                duration_sec=max(1.0, args.duration),
                source_fps=max(0.1, args.source_fps),
            )
            print(json.dumps(asdict(sample), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
