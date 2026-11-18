"""Tkinter UI for discovering, inspecting, and recording all visible LSL streams."""
from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from tkinter import END, BooleanVar, StringVar, Tk, messagebox, ttk
from typing import Any

from src.acquisition.cli import connect, resolve_streams
from src.acquisition.pipeline import extract_stream_metadata

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "configs" / "eeg_config.json").read_text(encoding="utf-8"))
EXPECTED_CHANNELS = list(CONFIG["eeg_channels"])


class StreamMonitorUI:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Mitsar LSL Stream Monitor - Device 1")
        self.root.geometry("1380x820")
        self.root.minsize(1000, 650)
        self.root.configure(bg="#0a111d")
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.streams: list[Any] = []
        self.metadata: dict[int, Any] = {}
        self.inspecting = False
        self.record_process: subprocess.Popen[str] | None = None
        self.stop_file: Path | None = None
        self.closing = False
        self.root_closed = False
        self.wave_stop_event: threading.Event | None = None
        self.wave_stream_index: int | None = None
        self.waveform_rows: deque[list[float]] = deque(maxlen=2500)
        self.waveform_labels: list[str] = []
        self.db_watch_stop: threading.Event | None = None
        self.db_session_id: str | None = None
        self.db_status_var = StringVar(value="Aura: not checked")

        self.status_var = StringVar(value="Ready. Start EEGStudio's LSL Outlet, then refresh.")
        self.participant_id_var = StringVar()
        self.participant_name_var = StringVar()
        self.duration_var = StringVar(value="0")
        self.db_seconds_var = StringVar(value="1200")
        self.use_neo4j_var = BooleanVar(value=True)

        self._style()
        self._layout()
        self.root.protocol("WM_DELETE_WINDOW", self.close_window)
        self.root.after(120, self._drain_events)
        self.root.after(250, self.discover)

    def _style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#0a111d")
        style.configure("TLabel", background="#0a111d", foreground="#dbe5f0")
        style.configure("TButton", background="#19283c", foreground="#dbe5f0",
                        padding=(10, 7), font=("Segoe UI", 9))
        style.map("TButton", background=[("active", "#263a53"), ("disabled", "#182333")])
        style.configure("TEntry", fieldbackground="#111d2c", foreground="#e1eaf5",
                        insertcolor="#e1eaf5", padding=6)
        style.configure("TCheckbutton", background="#0a111d", foreground="#a9b8cb")
        style.configure("Treeview", background="#0d1827", fieldbackground="#0d1827",
                        foreground="#dbe5f0", rowheight=31, borderwidth=0,
                        font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background="#172538", foreground="#93a8c2",
                        relief="flat", font=("Segoe UI", 8, "bold"))
        style.map("Treeview", background=[("selected", "#17372f")],
                  foreground=[("selected", "#edfff8")])
        style.configure("TCombobox", fieldbackground="#111d2c", background="#1b2a3f",
                        foreground="#dbe5f0")
        style.configure("Horizontal.TProgressbar", troughcolor="#19273a",
                        background="#57d4b0", bordercolor="#19273a")

    def _layout(self) -> None:
        header = self._frame(self.root, padding=(22, 18, 22, 12))
        header.pack(fill="x")
        self._label(header, "DEVICE 1 / PYTHON LSL ACQUISITION", 9, "#7189a7", bold=True).pack(anchor="w")
        self._label(header, "LSL Stream Monitor", 23, "#e9f1f8", bold=True).pack(anchor="w", pady=(5, 1))
        self._label(header, "See every resolved stream together, inspect connected channel metadata, and record the selected outlet.",
                    10, "#8191a7").pack(anchor="w")

        controls = self._frame(self.root, padding=(22, 4, 22, 10))
        controls.pack(fill="x")
        self.refresh_button = self._button(controls, "Refresh streams", self.discover, primary=True)
        self.refresh_button.pack(side="left", padx=(0, 8))
        self.inspect_button = self._button(controls, "Inspect selected", self.inspect_selected)
        self.inspect_button.pack(side="left", padx=(0, 8))
        self.inspect_all_button = self._button(controls, "Inspect all", self.inspect_all)
        self.inspect_all_button.pack(side="left")
        self.stream_count = self._label(controls, "0 streams", 9, "#8b9bb0")
        self.stream_count.pack(side="right", padx=(12, 0))
        self.busy = ttk.Progressbar(controls, mode="indeterminate", length=110,
                                    style="Horizontal.TProgressbar")
        self.busy.pack(side="right")

        table_frame = self._frame(self.root, padding=(22, 4, 22, 8))
        table_frame.pack(fill="both", expand=True)
        columns = ("index", "name", "type", "source", "uid", "channels", "nominal", "format", "labels")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        widths = {"index": 45, "name": 130, "type": 75, "source": 115, "uid": 205,
                  "channels": 75, "nominal": 95, "format": 85, "labels": 460}
        headings = {"index": "#", "name": "NAME", "type": "TYPE", "source": "SOURCE ID",
                    "uid": "STREAM UID", "channels": "CHANNELS", "nominal": "NOMINAL RATE",
                    "format": "FORMAT", "labels": "CHANNEL LABELS / INSPECTION STATUS"}
        for column in columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=widths[column], minwidth=45,
                              anchor="w" if column not in {"index", "channels", "nominal"} else "center",
                              stretch=column in {"uid", "labels"})
        vertical = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        self.table.bind("<<TreeviewSelect>>", self._selection_changed)

        detail_row = self._frame(self.root, padding=(22, 0, 22, 10))
        detail_row.pack(fill="x")
        self.detail_text = self._text(detail_row, height=5)
        self.detail_text.pack(fill="x")
        self._set_detail("Stream metadata will appear here. Resolver labels can be partial until an inlet connects.")

        graph_header = self._frame(self.root, padding=(22, 0, 22, 5))
        graph_header.pack(fill="x")
        self._label(graph_header, "LIVE EEG STREAM GRAPH", 9, "#70d9b9", bold=True).pack(side="left")
        self.preview_status = self._label(graph_header, "Preview stopped", 9, "#8b9bb0")
        self.preview_status.pack(side="right")
        self.preview_button = self._button(graph_header, "Start live preview", self.start_preview, primary=True)
        self.preview_button.pack(side="right", padx=(0, 12))
        self.preview_stop_button = self._button(graph_header, "Stop preview", self.stop_preview)
        self.preview_stop_button.pack(side="right", padx=(0, 8))
        self.preview_stop_button.configure(state="disabled")
        self._label(self.root, "Display only; incoming values are unchanged. Channel units and EEG labels remain unverified until inlet metadata provides them.",
                    8, "#8191a7").pack(anchor="w", padx=22, pady=(0, 4))
        self.wave_canvas = __import__("tkinter").Canvas(
            self.root, height=165, bg="#07111d", highlightthickness=1,
            highlightbackground="#26384d")
        self.wave_canvas.pack(fill="x", padx=22, pady=(0, 8))
        self.db_status_label = self._label(self.root, self.db_status_var.get(), 9, "#a9b8cb")
        self.db_status_label.pack(anchor="w", padx=22, pady=(0, 7))
        self.db_status_var.trace_add("write", lambda *_: self.db_status_label.configure(text=self.db_status_var.get()))
        self.root.after(250, self._draw_waveform)

        recorder = self._frame(self.root, padding=(22, 10, 22, 12), border=True)
        recorder.pack(fill="x", padx=22, pady=(0, 8))
        self._label(recorder, "RECORD SELECTED STREAM", 9, "#70d9b9", bold=True).grid(row=0, column=0, columnspan=8, sticky="w", pady=(0, 9))
        self._entry_group(recorder, "Participant code *", self.participant_id_var, 20, 1, 0)
        self._entry_group(recorder, "Display name (optional)", self.participant_name_var, 25, 1, 2)
        self._entry_group(recorder, "Duration seconds (0 = manual stop)", self.duration_var, 18, 1, 4)
        self._entry_group(recorder, "Aura raw seconds cap", self.db_seconds_var, 15, 1, 6)
        ttk.Checkbutton(recorder, text="Store session and raw chunks in Neo4j Aura",
                        variable=self.use_neo4j_var).grid(row=2, column=0, columnspan=4,
                                                         sticky="w", pady=(9, 0))
        self.record_button = self._button(recorder, "Start recording", self.start_recording, primary=True)
        self.record_button.grid(row=2, column=6, sticky="e", pady=(7, 0))
        self.stop_button = self._button(recorder, "Stop and flush", self.stop_recording, danger=True)
        self.stop_button.grid(row=2, column=7, sticky="e", padx=(8, 0), pady=(7, 0))
        self.stop_button.configure(state="disabled")
        for col in range(8):
            recorder.grid_columnconfigure(col, weight=1 if col % 2 == 1 else 0)

        self._label(self.root, "RECORDING LOG", 8, "#7189a7", bold=True).pack(anchor="w", padx=22, pady=(2, 5))
        log_frame = self._frame(self.root, padding=(22, 0, 22, 8))
        log_frame.pack(fill="both")
        self.log_text = self._text(log_frame, height=6)
        self.log_text.pack(fill="both", expand=True)
        self.status_label = self._label(self.root, "", 9, "#aab9ca")
        self.status_label.pack(fill="x", padx=22, pady=(0, 14))
        self.status_var.trace_add("write", lambda *_: self.status_label.configure(text=self.status_var.get()))

    def _frame(self, parent: Any, padding=(0, 0, 0, 0), border: bool = False):
        frame = ttk.Frame(parent, padding=padding)
        if border:
            frame.configure(relief="solid", borderwidth=1)
        return frame

    @staticmethod
    def _label(parent: Any, text: str, size: int, color: str, bold: bool = False):
        return ttk.Label(parent, text=text, foreground=color,
                         font=("Segoe UI", size, "bold" if bold else "normal"))

    @staticmethod
    def _button(parent: Any, text: str, command: Any, primary: bool = False, danger: bool = False):
        button = ttk.Button(parent, text=text, command=command)
        if primary:
            button.configure(style="Accent.TButton")
        elif danger:
            button.configure(style="Danger.TButton")
        return button

    def _entry_group(self, parent: Any, label: str, variable: StringVar, width: int, row: int, column: int) -> None:
        holder = ttk.Frame(parent)
        holder.grid(row=row, column=column, columnspan=2, sticky="ew", padx=(0, 10))
        self._label(holder, label, 8, "#8393a8").pack(anchor="w")
        entry = ttk.Entry(holder, textvariable=variable, width=width)
        entry.pack(fill="x", pady=(4, 0))

    @staticmethod
    def _text(parent: Any, height: int):
        from tkinter import Text
        return Text(parent, height=height, wrap="word", bg="#0d1827", fg="#b9c8d8",
                    insertbackground="#e8f1fa", relief="flat", padx=10, pady=8,
                    font=("Consolas", 9), state="disabled")

    @staticmethod
    def _write_text(widget: Any, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", END)
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _set_detail(self, text: str) -> None:
        self._write_text(self.detail_text, text)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(END, text.rstrip() + "\n")
        self.log_text.see(END)
        # Keep a bounded UI log for long recordings.
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 250:
            self.log_text.delete("1.0", f"{lines - 250}.0")
        self.log_text.configure(state="disabled")

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        if busy:
            self.busy.start(12)
        else:
            self.busy.stop()
        self.refresh_button.configure(state="disabled" if busy or self.record_process else "normal")
        self.inspect_button.configure(state="disabled" if busy or self.record_process else "normal")
        self.inspect_all_button.configure(state="disabled" if busy or self.record_process else "normal")
        if status:
            self.status_var.set(status)

    def discover(self) -> None:
        if self.inspecting or self.record_process:
            return
        self._set_busy(True, "Searching for all visible LSL streams...")
        def worker():
            try:
                streams = resolve_streams(wait_seconds=2.0)
                rows = [extract_stream_metadata(info) for info in streams]
                self.events.put(("discovered", (streams, rows)))
            except Exception as exc:
                self.events.put(("error", f"Stream discovery failed: {exc}"))
        threading.Thread(target=worker, name="lsl-discovery", daemon=True).start()

    def _render_streams(self, rows: list[Any]) -> None:
        for item in self.table.get_children():
            self.table.delete(item)
        self.metadata = {}
        for index, meta in enumerate(rows):
            self.metadata[index] = meta
            labels = ", ".join(label for label in meta.channel_labels if label) or "Not available until inlet inspection"
            self.table.insert("", END, iid=str(index), values=(
                index, meta.name or "", meta.stream_type or "", meta.source_id or "", meta.stream_id or "",
                meta.channel_count, f"{meta.nominal_srate:g} Hz", meta.channel_format or "", labels))
        self.stream_count.configure(text=f"{len(rows)} stream entr{'y' if len(rows) == 1 else 'ies'}")
        if not rows:
            self.status_var.set("No LSL stream found. Start EEGStudio's LSL Outlet and refresh.")
        else:
            duplicate_ids = {}
            source_counts = {}
            for index, meta in enumerate(rows):
                if meta.source_id:
                    source_counts[meta.source_id] = source_counts.get(meta.source_id, 0) + 1
                if meta.stream_id:
                    duplicate_ids.setdefault((meta.source_id, meta.stream_id), []).append(index)
            duplicate_count = sum(max(0, len(indices) - 1) for indices in duplicate_ids.values())
            suffix = f" {duplicate_count} duplicate resolver entr{'y' if duplicate_count == 1 else 'ies'} share an identical source/UID." if duplicate_count else ""
            if not duplicate_count and any(count > 1 for count in source_counts.values()):
                suffix = " Multiple entries share a source ID; compare UIDs and inlet labels before selecting."
            self.status_var.set(f"Found {len(rows)} LSL stream entr{'y' if len(rows) == 1 else 'ies'}.{suffix} Inspect labels before recording.")

    def _selected_index(self) -> int | None:
        selection = self.table.selection()
        return int(selection[0]) if selection else None

    def _selection_changed(self, _event: Any = None) -> None:
        index = self._selected_index()
        if index is None or index not in self.metadata:
            return
        self._show_metadata(index, self.metadata[index], inspected=False)

    def _show_metadata(self, index: int, meta: Any, inspected: bool) -> None:
        label_text = ", ".join(meta.channel_labels) if meta.channel_labels else "No channel labels exposed by this metadata."
        expected_lower = {label.casefold() for label in meta.channel_labels if label}
        missing = [name for name in EXPECTED_CHANNELS if name.casefold() not in expected_lower]
        additional = [name for name in meta.channel_labels if name and name.casefold() not in {c.casefold() for c in EXPECTED_CHANNELS}]
        status = "Connected inlet metadata" if inspected else "Resolver metadata (may be partial)"
        text = (f"Stream entry: {index}\nMetadata source: {status}\nName/type: {meta.name} / {meta.stream_type}\n"
                f"Source ID: {meta.source_id}\nStream UID: {meta.stream_id}\nChannels: {meta.channel_count}\n"
                f"Nominal rate: {meta.nominal_srate:g} Hz\nChannel format: {meta.channel_format}\n"
                f"Expected 19 EEG channels: {'all found' if not missing else 'missing ' + ', '.join(missing)}\n"
                f"Additional labels: {', '.join(additional) if additional else 'none or not available'}\n"
                f"Labels in stream order: {label_text}")
        self._set_detail(text)

    def inspect_selected(self) -> None:
        index = self._selected_index()
        if index is None:
            messagebox.showinfo("Select a stream", "Select a stream row first.")
            return
        self._inspect_indices([index])

    def inspect_all(self) -> None:
        if not self.streams:
            self.status_var.set("Refresh streams before inspecting.")
            return
        self._inspect_indices(list(range(len(self.streams))))

    def _inspect_indices(self, indices: list[int]) -> None:
        if self.inspecting or self.record_process:
            return
        self.inspecting = True
        self._set_busy(True, f"Opening inlet metadata for {len(indices)} stream entr{'y' if len(indices) == 1 else 'ies'}...")
        def worker():
            for index in indices:
                inlet = None
                try:
                    inlet = connect(self.streams[index], timeout=6.0)
                    meta = extract_stream_metadata(inlet.info())
                    self.events.put(("inspected", (index, meta)))
                except Exception as exc:
                    self.events.put(("inspect-error", (index, str(exc))))
                finally:
                    if inlet is not None:
                        try:
                            inlet.close_stream()
                        except Exception:
                            pass
            self.events.put(("inspect-complete", None))
        threading.Thread(target=worker, name="lsl-inspection", daemon=True).start()

    def start_recording(self) -> None:
        index = self._selected_index()
        if index is None:
            messagebox.showinfo("Select a stream", "Select and inspect an LSL stream before recording.")
            return
        meta = self.metadata.get(index)
        if meta is None or not meta.stream_id:
            messagebox.showerror("Stream UID missing", "Refresh and inspect this stream; no stable LSL UID is available.")
            return
        if not self.participant_id_var.get().strip():
            messagebox.showerror("Participant code required", "Enter the participant ID/code before recording.")
            return
        try:
            duration = int(self.duration_var.get().strip() or "0")
            db_seconds = int(self.db_seconds_var.get().strip() or "0")
        except ValueError:
            messagebox.showerror("Invalid duration", "Duration and Aura storage cap must be whole numbers of seconds.")
            return
        if duration < 0 or (self.use_neo4j_var.get() and db_seconds < 1):
            messagebox.showerror("Invalid duration", "Duration must be 0 or greater; Aura storage seconds must be positive.")
            return
        missing = [name for name in EXPECTED_CHANNELS
                   if name.casefold() not in {c.casefold() for c in meta.channel_labels if c}]
        if missing:
            proceed = messagebox.askyesno("EEG labels not verified",
                "These trained-model channels were not found in the inspected stream:\n\n" +
                ", ".join(missing) +
                "\n\nRaw recording can continue, but Device 2 will refuse model inference until the channel labels are corrected. Continue recording?")
            if not proceed:
                return

        # Keep a separate, read-only LSL inlet for the oscilloscope preview.
        if self.wave_stop_event is None:
            self.start_preview()

        control_dir = ROOT / "data" / "live" / "device1" / "control"
        control_dir.mkdir(parents=True, exist_ok=True)
        self.stop_file = control_dir / f"stop_{uuid.uuid4().hex}.flag"
        if self.stop_file.exists():
            self.stop_file.unlink()
        args = [sys.executable, "-m", "src.acquisition.cli", "record",
                "--stream-uid", meta.stream_id,
                "--participant-id", self.participant_id_var.get().strip(),
                "--output", str(ROOT / "data" / "live" / "device1"),
                "--stop-file", str(self.stop_file)]
        participant_name = self.participant_name_var.get().strip()
        if participant_name:
            args.extend(["--participant-name", participant_name])
        if duration > 0:
            args.extend(["--duration", str(duration)])
        if self.use_neo4j_var.get():
            args.extend(["--neo4j", "--db-raw-seconds", str(db_seconds)])
        try:
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            self.record_process = subprocess.Popen(
                args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=creationflags)
        except Exception as exc:
            self.record_process = None
            self.status_var.set(f"Could not start recording: {exc}")
            messagebox.showerror("Recorder failed to start", str(exc))
            return
        self.record_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.refresh_button.configure(state="disabled")
        self.inspect_button.configure(state="disabled")
        self.inspect_all_button.configure(state="disabled")
        self.status_var.set(f"Recording stream UID {meta.stream_id}. Duration: {'manual stop' if duration == 0 else f'{duration} seconds'}.")
        self._append_log("Starting selected LSL stream recorder...")
        self.db_status_var.set("Aura write/readback pending; waiting for recorder session ID.") if self.use_neo4j_var.get() else self.db_status_var.set("Aura storage disabled; recording locally only.")
        threading.Thread(target=self._read_record_log, args=(self.record_process,),
                         name="recording-log-reader", daemon=True).start()

    def start_preview(self) -> None:
        index = self._selected_index()
        if index is None or index >= len(self.streams):
            messagebox.showinfo("Select a stream", "Select an LSL stream row first.")
            return
        self.stop_preview()
        self.waveform_rows.clear()
        self.waveform_labels = []
        stop_event = threading.Event()
        self.wave_stop_event = stop_event
        self.wave_stream_index = index
        self.preview_button.configure(state="disabled")
        self.preview_stop_button.configure(state="normal")
        self.preview_status.configure(text="Connecting preview inlet…")
        info = self.streams[index]

        def worker() -> None:
            inlet = None
            try:
                inlet = connect(info, timeout=6.0)
                actual_meta = extract_stream_metadata(inlet.info())
                self.events.put(("wave-labels", actual_meta.channel_labels))
                self.events.put(("wave-status", f"Preview connected · {actual_meta.channel_count} channels · nominal {actual_meta.nominal_srate:g} Hz · units unknown"))
                while not stop_event.is_set():
                    samples, _timestamps = inlet.pull_chunk(timeout=0.25, max_samples=512)
                    if not samples:
                        continue
                    rows = []
                    for sample in samples:
                        try:
                            rows.append([float(value) for value in sample])
                        except (TypeError, ValueError):
                            continue
                    if rows:
                        self.events.put(("wave-data", rows))
            except Exception as exc:
                self.events.put(("wave-status", f"Preview connection error: {exc}"))
            finally:
                if inlet is not None:
                    try:
                        inlet.close_stream()
                    except Exception:
                        pass
                self.events.put(("wave-stopped", None))

        threading.Thread(target=worker, name="lsl-waveform-preview", daemon=True).start()

    def stop_preview(self) -> None:
        if self.wave_stop_event is not None:
            self.wave_stop_event.set()
            self.wave_stop_event = None
        if hasattr(self, "preview_stop_button"):
            self.preview_stop_button.configure(state="disabled")

    def _draw_waveform(self) -> None:
        if self.root_closed:
            return
        canvas = self.wave_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 600)
        height = max(canvas.winfo_height(), 120)
        rows = list(self.waveform_rows)
        if not rows:
            canvas.create_text(width / 2, height / 2, text="Start live preview to display incoming LSL samples",
                               fill="#8191a7", font=("Segoe UI", 10))
        else:
            channel_count = min(len(rows[-1]), 32)
            left, right, top, bottom = 70, width - 12, 8, height - 8
            plot_h = bottom - top
            for channel in range(channel_count):
                y0 = top + channel * plot_h / channel_count
                y1 = top + (channel + 1) * plot_h / channel_count
                center = (y0 + y1) / 2
                canvas.create_line(left, center, right, center, fill="#17283a")
                segment = [row[channel] for row in rows if len(row) > channel and math.isfinite(row[channel])]
                label = (self.waveform_labels[channel].strip() if channel < len(self.waveform_labels) else "")
                label = label or f"Ch {channel + 1:02d}"
                canvas.create_text(5, center, anchor="w", text=label[:10], fill="#87a0ba",
                                   font=("Consolas", 8))
                if len(segment) < 2:
                    continue
                low, high = min(segment), max(segment)
                span = max(high - low, 1e-12)
                usable = max(2.0, (y1 - y0) * 0.78)
                points = []
                for pos, value in enumerate(segment):
                    x = left + pos * (right - left) / max(1, len(segment) - 1)
                    y = y1 - 1 - ((value - low) / span) * usable
                    points.extend((x, y))
                if len(points) >= 4:
                    canvas.create_line(*points, fill="#57d4b0", width=1)
            canvas.create_text(right, height - 2, anchor="se", text=f"latest {len(rows)} samples · display autoscaled",
                               fill="#7189a7", font=("Segoe UI", 7))
        self.root.after(250, self._draw_waveform)

    def _start_db_watch(self, session_id: str) -> None:
        if self.db_watch_stop is not None:
            self.db_watch_stop.set()
        stop_event = threading.Event()
        self.db_watch_stop = stop_event
        self.db_session_id = session_id
        self.db_status_var.set(f"Aura session {session_id}: checking stored chunks…")

        def worker() -> None:
            try:
                from src.storage.neo4j_store import Neo4jReader
                reader = Neo4jReader()
                cursor = 0
                total_samples = 0
                seen_chunks = 0
                try:
                    while not stop_event.is_set():
                        session_rows = reader.list_sessions(limit=200)
                        session_row = next((row for row in session_rows
                                            if row["session"].get("session_id") == session_id), None)
                        if session_row is None:
                            self.events.put(("db-status", "Aura connected, but this session is not visible in readback yet."))
                        else:
                            chunks = reader.read_sample_chunks_after(session_id, cursor, limit=100)
                            for chunk in chunks:
                                if chunk["chunk_index"] != cursor:
                                    raise ValueError(f"readback chunk sequence gap: expected {cursor}, got {chunk['chunk_index']}")
                                if any(len(sample) != len(chunk["channel_labels"]) for sample in chunk["samples"]):
                                    raise ValueError(f"chunk {cursor} channel width does not match its labels")
                                cursor += 1
                                seen_chunks += 1
                                total_samples += chunk["sample_count"]
                            stored_chunks = int(session_row["raw_chunk_count"])
                            stored_samples = int(session_row["raw_sample_count"])
                            self.events.put(("db-status", f"Aura readback verified · session {session_id[:8]} · {seen_chunks}/{stored_chunks} chunks fetched · {total_samples}/{stored_samples} samples checked"))
                            ended = bool(session_row["session"].get("ended_at_utc"))
                            if ended and seen_chunks >= stored_chunks:
                                break
                        stop_event.wait(1.0)
                finally:
                    reader.close()
            except Exception as exc:
                self.events.put(("db-status", f"Aura readback failed: {exc}"))

        threading.Thread(target=worker, name="neo4j-readback-watch", daemon=True).start()

    def _read_record_log(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self.events.put(("record-log", line.rstrip()))
        code = process.wait()
        self.events.put(("record-exit", code))

    def stop_recording(self) -> None:
        if not self.record_process or self.record_process.poll() is not None:
            return
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stop requested. Flushing local files and queued Aura chunks...")
        if self.stop_file:
            try:
                self.stop_file.write_text("stop\n", encoding="utf-8")
            except OSError as exc:
                messagebox.showerror("Cannot request graceful stop", str(exc))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "discovered":
                    streams, rows = payload
                    self.streams = streams
                    self.metadata = {i: row for i, row in enumerate(rows)}
                    self._render_streams(rows)
                    self._set_busy(False)
                elif kind == "inspected":
                    index, meta = payload
                    self.metadata[index] = meta
                    self.table.item(str(index), values=(index, meta.name, meta.stream_type,
                        meta.source_id, meta.stream_id, meta.channel_count,
                        f"{meta.nominal_srate:g} Hz", meta.channel_format,
                        ", ".join(meta.channel_labels) if meta.channel_labels else "No labels exposed"))
                    self._show_metadata(index, meta, inspected=True)
                    self.status_var.set(f"Inspected stream entry {index}; actual metadata shown.")
                elif kind == "inspect-error":
                    index, error = payload
                    self._append_log(f"Inspect stream [{index}] failed: {error}")
                elif kind == "inspect-complete":
                    self.inspecting = False
                    self._set_busy(False, "Stream inspection complete. Select a row to review its channel mapping.")
                elif kind == "record-log":
                    self._append_log(payload)
                    if payload.startswith("Session ID: ") and self.use_neo4j_var.get():
                        self._start_db_watch(payload.split(":", 1)[1].strip())
                elif kind == "wave-labels":
                    self.waveform_labels = payload
                elif kind == "wave-data":
                    self.waveform_rows.extend(payload)
                elif kind == "wave-status":
                    self.preview_status.configure(text=payload)
                elif kind == "wave-stopped":
                    self.preview_button.configure(state="normal")
                    self.preview_stop_button.configure(state="disabled")
                elif kind == "db-status":
                    self.db_status_var.set(payload)
                elif kind == "record-exit":
                    self.record_process = None
                    if self.stop_file and self.stop_file.exists():
                        self.stop_file.unlink(missing_ok=True)
                    self.stop_file = None
                    self.record_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.refresh_button.configure(state="normal")
                    self.inspect_button.configure(state="normal")
                    self.inspect_all_button.configure(state="normal")
                    self.stop_preview()
                    if self.use_neo4j_var.get() and self.db_session_id is None:
                        self.db_status_var.set("Aura not verified: recorder stopped before creating a database session. Check the log for the connection/driver error.")
                    self.status_var.set("Recording finished and local files were flushed." if payload == 0 else
                                        f"Recorder exited with code {payload}; review the log above.")
                    if self.closing:
                        self.root_closed = True
                        self.root.after_idle(self.root.destroy)
                elif kind == "error":
                    self._set_busy(False, payload)
                    self._append_log(payload)
                    messagebox.showerror("LSL error", payload)
        except queue.Empty:
            pass
        if not self.root_closed:
            self.root.after(120, self._drain_events)

    def close_window(self) -> None:
        if self.record_process and self.record_process.poll() is None:
            if not messagebox.askyesno("Recording is active", "Request a graceful recorder stop and close after files are flushed?"):
                return
            self.closing = True
            self.stop_recording()
            return
        self.stop_preview()
        if self.db_watch_stop is not None:
            self.db_watch_stop.set()
        self.root.destroy()


def main() -> None:
    root = Tk()
    app = StreamMonitorUI(root)
    style = ttk.Style()
    style.configure("Accent.TButton", background="#54d1ad", foreground="#09251f",
                    font=("Segoe UI", 9, "bold"), padding=(11, 7))
    style.map("Accent.TButton", background=[("active", "#75e2c2"), ("disabled", "#40534f")])
    style.configure("Danger.TButton", background="#3a2430", foreground="#ffc1c7",
                    font=("Segoe UI", 9, "bold"), padding=(11, 7))
    style.map("Danger.TButton", background=[("active", "#56303b"), ("disabled", "#2d2830")])
    root.mainloop()


if __name__ == "__main__":
    main()
