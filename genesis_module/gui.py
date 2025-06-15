import time
import threading
import curses
from collections import deque
from typing import Dict, Any
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
        length, step count, anchor bias statistics, memory usage information and
        amplifier/gate status flags.
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
    gpu_mb = None
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated(anchor_bias.device) / (1024 * 1024)
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
        "gpu_memory_mb": gpu_mb,
    }


def launch_gui(plugin_or_model, refresh: float = 1.0):
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
        Interval in seconds between screen updates. Defaults to ``1.0``.

    Returns
    -------
    threading.Event
        Event which can be set to terminate the dashboard.
    """

    stop_event = threading.Event()
    novelty_history = deque(maxlen=50)

    def _dashboard(stdscr):
        curses.curs_set(0)
        while not stop_event.is_set():
            metrics = get_gui_metrics(plugin_or_model)
            novelty_history.append(metrics["novelty_score"])
            hist = torch.histc(metrics["anchor_bias"], bins=10, min=-float(plugin_or_model.bias_max), max=float(plugin_or_model.bias_max))
            stdscr.erase()
            stdscr.addstr(0, 0, "GENESIS Dashboard")
            stdscr.addstr(2, 0, f"Replay buffer size: {metrics['replay_buffer_len']}")
            stdscr.addstr(3, 0, f"Avg priority: {metrics['avg_priority']:.3f}")
            stdscr.addstr(4, 0, f"Amplifier: {'ON' if metrics['amplifier_active'] else 'OFF'}  Gate: {'ON' if metrics['gate_active'] else 'OFF'}")
            stdscr.addstr(5, 0, f"Step: {metrics['step_count']}  Anchor μ={metrics['anchor_mean']:.3f} σ={metrics['anchor_std']:.3f}")
            mem_line = f"CPU MB: {metrics['cpu_memory_mb']:.1f}"
            if metrics['gpu_memory_mb'] is not None:
                mem_line += f"  GPU MB: {metrics['gpu_memory_mb']:.1f}"
            stdscr.addstr(6, 0, mem_line)
            stdscr.addstr(8, 0, "Anchor bias histogram:")
            for i, val in enumerate(hist.int().tolist()):
                stdscr.addstr(9 + i, 0, f"{i:02d}: " + '#' * int(val))
            stdscr.addstr(20, 0, "Novelty score history:")
            hist_vals = ", ".join(f"{v:.2f}" for v in list(novelty_history)[-10:])
            stdscr.addstr(21, 0, hist_vals)
            stdscr.refresh()
            time.sleep(refresh)

    thread = threading.Thread(target=lambda: curses.wrapper(_dashboard), daemon=True)
    thread.start()
    return stop_event
