"""Measure model capacity using repeated real windows, without official scores."""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from examples.b_real_data_smoke import _prepare
from ptop.schema import ContractError
from ptop.temporal import MODERN_MODEL_NAMES, ModelSpec, create_score_model
from ptop.training import resolve_device, seed_everything, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise ContractError("this benchmark requires an explicitly assigned CUDA device")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("scope") != "diagnostic":
        raise ContractError("use explicit real-bar diagnostic config; this does not measure official prediction quality")
    repository = Path(__file__).resolve().parents[1]
    if args.output.resolve() == repository or repository in args.output.resolve().parents:
        raise ContractError("keep real-window benchmark artifacts and cache outside the repository")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared = _prepare(config, output_dir=args.output.parent / "preparation")
    batches = list(prepared.validation_loader())
    batch = max(batches, key=lambda b: b.metadata.decision_time.max())
    eligible = np.flatnonzero(batch.predictable)
    if not len(eligible):
        raise ContractError("no observed real windows to profile")
    results = []
    for name in MODERN_MODEL_NAMES:
        seed_everything(7)
        spec = ModelSpec(name=name, encoder="ALSTM", lookback=prepared.window_length,
                         width=32, layers=2, pooling="mean", head_width=32)
        model = create_score_model(spec).to(device).eval()
        for size in (32, 256, 1024):
            indices = np.resize(eligible, size)
            inputs = [torch.as_tensor(value[indices], device=device)
                      for value in (batch.x, batch.observed, batch.step_valid)]
            with torch.inference_mode():
                for _ in range(3):
                    model(*inputs)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                elapsed = []
                for _ in range(10):
                    started = time.perf_counter()
                    score = model(*inputs)
                    torch.cuda.synchronize(device)
                    elapsed.append(time.perf_counter() - started)
                assert torch.isfinite(score).all()
            results.append({"model": spec.to_dict(), "batch_size": size, "lookback": prepared.window_length,
                            "median_forward_seconds": statistics.median(elapsed),
                            "samples_per_second": size / statistics.median(elapsed),
                            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)})
            del inputs, score
    write_json(args.output, {"scope": "capacity probe; repeated real windows and random weights; not official pool or performance",
                            "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
                            "real_source_windows": len(eligible), "preprocessor_fingerprint": prepared.processor.fingerprint,
                            "excludes": ["source materialization", "daily normalization", "tensor transfer", "output audit", "full platform runtime"],
                            "results": results})
    print(json.dumps({"output": str(args.output), "measured_configurations": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
