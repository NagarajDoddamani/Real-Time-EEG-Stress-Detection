"""Publish clearly synthetic 19-channel samples over LSL for integration checks."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHANNELS = json.loads((ROOT / "configs" / "eeg_config.json").read_text(encoding="utf-8"))["eeg_channels"]


def synthetic_chunk(start_sample: int, count: int, rate: float) -> np.ndarray:
    """Return deterministic arbitrary-unit signals; these are not physiological EEG."""
    samples = np.empty((count, len(CHANNELS)), dtype=np.float32)
    for offset in range(count):
        t = (start_sample + offset) / rate
        for channel_index in range(len(CHANNELS)):
            phase = channel_index * 0.17
            samples[offset, channel_index] = (
                8.0 * math.sin(2 * math.pi * 10.0 * t + phase)
                + 2.0 * math.sin(2 * math.pi * 20.0 * t + phase / 2)
                + 0.5 * math.sin(2 * math.pi * 3.0 * t)
            )
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0, help="wall-clock duration in seconds")
    parser.add_argument("--rate", type=float, default=500.0, help="published nominal rate; model expects 500 Hz")
    parser.add_argument("--chunk-samples", type=int, default=50)
    args = parser.parse_args(argv)
    if args.duration <= 0 or args.rate <= 0 or args.chunk_samples < 1:
        parser.error("duration, rate, and chunk-samples must be positive")
    try:
        from pylsl import StreamInfo, StreamOutlet, cf_float32, local_clock
    except ImportError as exc:
        parser.exit(2, f"pylsl unavailable: {exc}\n")

    info = StreamInfo("SIMULATED_EEG_19CH_NOT_MITSAR", "EEG", len(CHANNELS), args.rate,
                      cf_float32, "SIMULATED_EEG_NOT_MITSAR_001")
    channel_node = info.desc().append_child("channels")
    for label in CHANNELS:
        channel_node.append_child("channel").append_child_value("label", label).append_child_value(
            "unit", "arbitrary synthetic units")
    info.desc().append_child_value("source", "SIMULATED ONLY - not Mitsar hardware")
    outlet = StreamOutlet(info, max_buffered=60)
    print(f"Publishing SIMULATED data only: {len(CHANNELS)} channels at {args.rate:g} Hz. "
          "This is not physiological EEG and not a device test.", flush=True)
    chunk_count = 0
    sent_samples = 0
    first_clock = local_clock()
    stop_at = time.monotonic() + args.duration
    try:
        while time.monotonic() < stop_at:
            count = min(args.chunk_samples, max(1, int(math.ceil((stop_at - time.monotonic()) * args.rate))))
            values = synthetic_chunk(sent_samples, count, args.rate)
            timestamp0 = first_clock + sent_samples / args.rate
            timestamps = [timestamp0 + i / args.rate for i in range(count)]
            outlet.push_chunk(values, timestamps, pushthrough=True)
            sent_samples += count
            chunk_count += 1
            target_time = first_clock + sent_samples / args.rate
            wait = target_time - local_clock()
            if wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        print("Synthetic publisher stop requested.", flush=True)
    print(f"Synthetic publisher sent {sent_samples} samples in {chunk_count} chunks.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
