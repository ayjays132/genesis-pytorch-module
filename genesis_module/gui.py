import time
import threading
import curses
from collections import deque
from typing import Dict, Any
import torch


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
        length and basic sampling statistics, and amplifier/gate status flags.
    """
    anchor_bias = obj.anchor_bias.detach().cpu()
    novelty = float(obj.novelty_score.detach().cpu())
    buf_len = len(obj.replay_buffer.buffer)
    priorities = [p for (_, _, p) in obj.replay_buffer.buffer]
    avg_priority = float(torch.tensor(priorities).mean().item()) if priorities else 0.0
    amp_active = bool(obj.importance_scores.sum().item() > 0)
    gate_active = bool(getattr(obj, "gate", None) is not None)
    return {
        "anchor_bias": anchor_bias,
        "novelty_score": novelty,
        "replay_buffer_len": buf_len,
        "avg_priority": avg_priority,
        "amplifier_active": amp_active,
        "gate_active": gate_active,
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
            stdscr.addstr(6, 0, "Anchor bias histogram:")
            for i, val in enumerate(hist.int().tolist()):
                stdscr.addstr(7 + i, 0, f"{i:02d}: " + '#' * int(val))
            stdscr.addstr(18, 0, "Novelty score history:")
            hist_vals = ", ".join(f"{v:.2f}" for v in list(novelty_history)[-10:])
            stdscr.addstr(19, 0, hist_vals)
            stdscr.refresh()
            time.sleep(refresh)

    thread = threading.Thread(target=lambda: curses.wrapper(_dashboard), daemon=True)
    thread.start()
    return stop_event
