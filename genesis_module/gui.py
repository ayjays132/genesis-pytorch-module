"""Simple curses-based dashboard for monitoring GENESIS training metrics."""

import time
import threading
import curses
import os
from collections import deque
from typing import Any, Dict, Optional, Tuple
import torch
import psutil


def get_gui_metrics(obj) -> Dict[str, Any]:
    """Collect metrics from a GENESIS module or plugin.

    Parameters
    ----------
    obj : IntegratedLearningModule or GenesisPlugin
        The model/plugin instance providing GENESIS functionality.

    Returns
    -------
    dict
        Dictionary containing anchor bias tensor, novelty score, replay buffer
        length, step count, anchor bias statistics, memory usage information,
        CPU/GPU load, and amplifier/gate status flags.
    """
    anchor_bias = obj.anchor_bias.detach().cpu()
    novelty = float(obj.novelty_score.detach().cpu())
    buf_len = len(obj.replay_buffer.buffer)
    priorities = [p for (_, _, p) in obj.replay_buffer.buffer]
    avg_priority = float(torch.tensor(priorities).mean().item()) if priorities else 0.0
    amp_active = bool(obj.importance_scores.sum().item() > 0)
    gate_active = bool(getattr(obj, "gate", None) is not None)
    step_count = int(obj.steps.item())
    anchor_mean = float(anchor_bias.mean().item())
    anchor_std = float(anchor_bias.std().item())
    process = psutil.Process()
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent()
    gpu_mb = None
    gpu_util = None
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated(anchor_bias.device) / (1024 * 1024)
        try:
            gpu_util = torch.cuda.utilization()
        except Exception:
            gpu_util = None
    return {
        "anchor_bias": anchor_bias,
        "novelty_score": novelty,
        "replay_buffer_len": buf_len,
        "avg_priority": avg_priority,
        "amplifier_active": amp_active,
        "gate_active": gate_active,
        "step_count": step_count,
        "anchor_mean": anchor_mean,
        "anchor_std": anchor_std,
        "cpu_memory_mb": mem_mb,
        "cpu_percent": cpu_percent,
        "gpu_memory_mb": gpu_mb,
        "gpu_utilization": gpu_util,
    }


def launch_gui(
    plugin_or_model,
    refresh: Optional[float] = None,
) -> Tuple[threading.Event, threading.Thread]:
    """Launch a simple curses dashboard visualizing GENESIS metrics.

    The dashboard runs in a separate daemon thread and updates every ``refresh``
    seconds. ``plugin_or_model`` should be an instance of ``IntegratedLearningModule``
    or ``GenesisPlugin`` used during training.

    Parameters
    ----------
    plugin_or_model : object
        Model providing ``anchor_bias``, ``replay_buffer``, ``novelty_score`` and
        ``importance_scores`` attributes.
    refresh : float, optional
        Interval in seconds between screen updates. If ``None``, the value is
        read from the ``GENESIS_GUI_REFRESH`` environment variable or defaults
        to ``1.0``.

    Returns
    -------
    (threading.Event, threading.Thread)
        Event which can be set to terminate the dashboard and the dashboard
        thread object so it can be joined.
    """

    if refresh is None:
        env_val = os.getenv("GENESIS_GUI_REFRESH")
        try:
            refresh = float(env_val) if env_val is not None else 1.0
        except ValueError:
            refresh = 1.0

    stop_event = threading.Event()
    novelty_history = deque(maxlen=50)

    def _dashboard(stdscr):
        curses.curs_set(0)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        while not stop_event.is_set():
            metrics = get_gui_metrics(plugin_or_model)
            novelty_history.append(metrics["novelty_score"])
            hist = torch.histc(
                metrics["anchor_bias"],
                bins=10,
                min=-float(plugin_or_model.bias_max),
                max=float(plugin_or_model.bias_max),
            )
            stdscr.erase()
            stdscr.addstr(0, 0, "GENESIS Dashboard", curses.color_pair(1))
            stdscr.addstr(2, 0, f"Replay buffer size: {metrics['replay_buffer_len']}")
            stdscr.addstr(3, 0, f"Avg priority: {metrics['avg_priority']:.3f}")
            stdscr.addstr(
                4,
                0,
                f"Amplifier: {'ON' if metrics['amplifier_active'] else 'OFF'}  "
                f"Gate: {'ON' if metrics['gate_active'] else 'OFF'}",
            )
            stdscr.addstr(
                5,
                0,
                f"Step: {metrics['step_count']}  "
                f"Anchor μ={metrics['anchor_mean']:.3f} σ={metrics['anchor_std']:.3f}",
            )
            nov_color = curses.color_pair(2) if metrics['novelty_score'] > 0.8 else curses.A_NORMAL
            stdscr.addstr(6, 0, f"Novelty score: {metrics['novelty_score']:.3f}", nov_color)
            mem_line = (
                f"CPU MB: {metrics['cpu_memory_mb']:.1f}  CPU%: {metrics['cpu_percent']:.1f}"
            )
            if metrics['gpu_memory_mb'] is not None:
                mem_line += f"  GPU MB: {metrics['gpu_memory_mb']:.1f}"
            if metrics['gpu_utilization'] is not None:
                mem_line += f" GPU%: {metrics['gpu_utilization']:.1f}"
            mem_warn = (
                metrics['cpu_memory_mb'] > 1024
                or metrics['cpu_percent'] > 90
                or (metrics['gpu_memory_mb'] and metrics['gpu_memory_mb'] > 1024)
                or (metrics['gpu_utilization'] and metrics['gpu_utilization'] > 90)
            )
            color = curses.color_pair(2) if mem_warn else curses.A_NORMAL
            stdscr.addstr(7, 0, mem_line, color)
            stdscr.addstr(9, 0, "Anchor bias histogram:")
            for i, val in enumerate(hist.int().tolist()):
                stdscr.addstr(10 + i, 0, f"{i:02d}: " + '#' * int(val))
            stdscr.addstr(21, 0, "Novelty score history:")
            hist_vals = ", ".join(f"{v:.2f}" for v in list(novelty_history)[-10:])
            stdscr.addstr(22, 0, hist_vals)
            stdscr.refresh()
            time.sleep(refresh)

    thread = threading.Thread(target=lambda: curses.wrapper(_dashboard), daemon=True)
    thread.start()
    return stop_event, thread
