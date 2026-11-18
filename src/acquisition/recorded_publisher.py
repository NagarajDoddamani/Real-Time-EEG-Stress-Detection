"""Replay one existing raw CSV through LSL at its configured 500 Hz rate."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "raw" / "Chaitanya.csv"
CONFIG = json.loads((ROOT / "configs" / "eeg_config.json").read_text(encoding="utf-8"))
CHANNELS = list(CONFIG["eeg_channels"])


def load_recording(path: Path) -> tuple[np.ndarray, str, int]:
    """Use the project's header-aware loader; keep raw EEG values unchanged."""
    from src.training.predict_all_recordings import load_raw_recording
    return load_raw_recording(path, CHANNELS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="one existing raw CSV; headers are mapped to the configured EEG channels")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="replay prefix duration; capped at source recording length")
    parser.add_argument("--chunk-samples", type=int, default=50)
    parser.add_argument("--startup-delay", type=float, default=8.0,
                        help="seconds to keep the outlet available before beginning sample timestamps")
    args = parser.parse_args(argv)
    fs = float(CONFIG["sampling_frequency_hz"])
    if not CONFIG.get("sampling_frequency_verified") or fs != 500:
        parser.error("recorded replay requires the project's user-verified 500 Hz rate")
    if args.duration <= 0 or args.chunk_samples < 1 or args.startup_delay < 0:
        parser.error("duration and chunk-samples must be positive; startup-delay cannot be negative")
    try:
        from pylsl import StreamInfo, StreamOutlet, cf_double64, local_clock
    except ImportError as exc:
        parser.exit(2, f"pylsl unavailable: {exc}\n")
    try:
        data, delimiter, row_width_issues = load_recording(args.input)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Replay input rejected: {exc}\n")
    sample_limit = min(len(data), int(math.floor(args.duration * fs)))
    if sample_limit < 1:
        parser.error("input contains no samples in the requested replay duration")
    info = StreamInfo("RECORDED_CSV_REPLAY_NOT_LIVE_EEG", "EEG", len(CHANNELS), fs,
                      cf_double64, "RECORDED_CSV_REPLAY_NOT_HARDWARE")
    channels_node = info.desc().append_child("channels")
    for label in CHANNELS:
        channels_node.append_child("channel").append_child_value("label", label).append_child_value(
            "unit", "source units unverified")
    info.desc().append_child_value("source", "RECORDED CSV REPLAY - not live EEG")
    info.desc().append_child_value("source_filename", args.input.name)
    outlet = StreamOutlet(info, max_buffered=60)
    print(f"Replaying {args.input.name} as recorded CSV data (delimiter={delimiter}; "
          f"recovered row-width issues={row_width_issues}); channels={len(CHANNELS)}, "
          f"rate={fs:g} Hz, samples={sample_limit}. Not live Mitsar EEG. "
          f"Starting sample transmission in {args.startup_delay:g} seconds.", flush=True)
    if args.startup_delay:
        time.sleep(args.startup_delay)
    start_clock = local_clock()
    index = 0
    try:
        while index < sample_limit:
            end = min(index + args.chunk_samples, sample_limit)
            timestamps = [start_clock + sample_index / fs for sample_index in range(index, end)]
            outlet.push_chunk(np.asarray(data[index:end], dtype=np.float64), timestamps, pushthrough=True)
            index = end
            wait = (start_clock + index / fs) - local_clock()
            if wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        print("Replay interrupted by user.", flush=True)
    print(f"Recorded-data replay ended after publishing {index} samples ({index/fs:.3f} seconds).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
