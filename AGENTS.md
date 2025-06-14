# AGENT Instructions

This repository contains **GENESIS** (Grounded Ethical Neuromodulated Exponential Self-Improving System), a PyTorch module intended for human-like learning. Key features include a self-replay buffer, sticky learning amplifier, ethical gating, and a persistence layer. The design is inspired by principles from infant cognitive development and aims for memory efficiency and safe behavior.

## Contribution guidelines

- **Maintain core functionality.** Any changes should keep the replay buffer, sticky learning amplifier, ethical gate, and persistence mechanisms functional.
- **Follow the README.** The detailed design philosophy in `README.md` explains the learning mechanisms and safety plan. Contributions should respect these ideas.
- **Write tests.** Update or extend `test_genesis.py` when modifying the code. Tests should cover initialization, ethical gating, training behavior, and replay buffer usage.
- **Run tests locally.** Install PyTorch (version `>=1.9`) and execute `python test_genesis.py` before committing. Ensure the tests pass.
- **Code style.** Use Python 3.8+ syntax and keep functions well documented with inline comments when needed.
- **Safety first.** The ethical gate and other safety mechanisms must not be removed. Refer to the safety plan for context.

