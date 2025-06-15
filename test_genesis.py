import torch
import torch.nn as nn
import pytest
from genesis_module import (
    IntegratedLearningModule,
    SelfReplayBuffer,
    EthicalGate,
    GenesisPlugin,
    attach_genesis_plugin,
)


class StepCounterOptimizer(torch.optim.Adam):
    """Optimizer that counts how many times step() and zero_grad() are called."""

    def __init__(self, params, lr=0.001):
        super().__init__(params, lr=lr)
        self.step_calls = 0
        self.zero_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)

    def zero_grad(self, set_to_none: bool = False):
        self.zero_calls += 1
        return super().zero_grad(set_to_none=set_to_none)


def test_genesis_module():
    print("Starting GENESIS module test...")

    # Test 1: Initialization and basic forward pass
    print("\nTest 1: Initialization and basic forward pass")
    input_dim = 50
    hidden_dim = 128
    output_dim = 100
    vocab_size = output_dim
    disallowed = [0, 1, 2]

    model = IntegratedLearningModule(
        input_dim, hidden_dim, output_dim, vocab_size=vocab_size, disallowed_tokens=disallowed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model initialized and moved to {device}.")

    x_batch = torch.randn(4, 10, input_dim).to(device)  # Batch size 4, seq_len 10
    logits, raw_logits, final_hidden = model(x_batch)

    assert logits.shape == torch.Size([4, output_dim]), f"Expected logits shape [4, {output_dim}], got {logits.shape}"
    assert raw_logits.shape == torch.Size(
        [4, output_dim]
    ), f"Expected raw_logits shape [4, {output_dim}], got {raw_logits.shape}"
    assert final_hidden.shape == torch.Size(
        [4, hidden_dim]
    ), f"Expected final_hidden shape [4, {hidden_dim}], got {final_hidden.shape}"
    print("Basic forward pass successful. Shapes are correct.")

    # Test 2: Ethical Gating
    print("\nTest 2: Ethical Gating")
    # Check if disallowed tokens have very low probability after filtering
    # For simplicity, we check if the logits for disallowed tokens are indeed -1e9
    disallowed_indices = torch.tensor(disallowed).to(device)
    assert torch.all(logits[:, disallowed_indices] < -1e8), "Ethical gate did not filter disallowed tokens correctly."
    print("Ethical gating successful: Disallowed tokens are suppressed.")

    # Test 3: Training step with replay and amplifier
    print("\nTest 3: Training step with replay and amplifier")
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()

    initial_anchor_bias = model.anchor_bias.clone().detach()
    initial_anchor_bias_ref = model.anchor_bias_ref.clone().detach()
    initial_novelty_score = model.novelty_score.clone().detach()

    # Run a few training steps to observe changes and capture learning rate update
    initial_lr = optim.param_groups[0]["lr"]
    lr_after_first = None
    for i in range(5):
        # Avoid disallowed tokens to keep the loss within a reasonable range
        y_batch = torch.randint(3, output_dim, (4,)).to(device)
        loss_val = model.training_step(x_batch, y_batch, optim, crit)
        if i == 0:
            lr_after_first = optim.param_groups[0]["lr"]
        print(f"  Step {i+1}, Loss: {loss_val:.4f}, Novelty Score: {model.novelty_score.item():.4f}")

    # Check if anchor bias changed and reference updated
    assert not torch.equal(initial_anchor_bias, model.anchor_bias), "Anchor bias did not change after training steps."
    if model.importance_scores.sum() > 0:
        assert not torch.equal(
            initial_anchor_bias_ref, model.anchor_bias_ref
        ), "Anchor bias reference did not update when amplifier triggered."
    assert not torch.equal(
        initial_novelty_score, model.novelty_score
    ), "Novelty score did not change after training steps."
    assert (
        lr_after_first is not None and lr_after_first != initial_lr
    ), "Learning rate was not updated by adaptive scheduler."
    print("Training steps successful: Anchor bias and novelty score updated.")

    # Test 4: Replay buffer functionality
    print("\nTest 4: Replay buffer functionality")
    # After several steps, the buffer should have some items
    assert len(model.replay_buffer.buffer) > 0, "Replay buffer is empty."
    print(f"Replay buffer contains {len(model.replay_buffer.buffer)} items.")

    # Sample from buffer and check shapes
    h_replay, t_replay = model.replay_buffer.sample(batch_size=1, device=device)
    assert h_replay is not None and t_replay is not None, "Failed to sample from replay buffer."
    assert h_replay.shape == torch.Size(
        [1, hidden_dim]
    ), f"Expected replay hidden shape [1, {hidden_dim}], got {h_replay.shape}"
    assert t_replay.shape == torch.Size([1]), f"Expected replay target shape [1], got {t_replay.shape}"
    # Ensure memory-efficient dtype storage and automatic float32 conversion
    assert model.replay_buffer.buffer[0][0].dtype == torch.float16, "Replay buffer should store float16 by default"
    assert h_replay.dtype == torch.float32, "Sampled hidden state should be float32"
    print("Replay buffer sampling successful.")

    print("\nAll GENESIS module tests passed successfully!")


def test_genesis_plugin():
    print("\nTesting GenesisPlugin integration")
    input_dim = 32
    hidden_dim = 64
    output_dim = 50
    disallowed = [0, 1]

    base = nn.Linear(input_dim, hidden_dim)
    plugin = GenesisPlugin(hidden_dim, output_dim, vocab_size=output_dim, disallowed_tokens=disallowed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = base.to(device)
    plugin = plugin.to(device)

    optimizer = torch.optim.Adam(list(base.parameters()) + list(plugin.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, input_dim).to(device)
    hidden = torch.relu(base(x))
    logits, _, _ = plugin(hidden)
    assert logits.shape == torch.Size([4, output_dim])

    y = torch.randint(0, output_dim, (4,)).to(device)
    initial_anchor_ref = plugin.anchor_bias_ref.clone().detach()
    loss_val = plugin.training_step(hidden, y, optimizer, criterion)
    assert len(plugin.replay_buffer.buffer) > 0
    if plugin.importance_scores.sum() > 0:
        assert not torch.equal(
            initial_anchor_ref, plugin.anchor_bias_ref
        ), "Plugin anchor bias reference did not update when amplifier triggered."
    print(f"GenesisPlugin training step loss: {loss_val:.4f}")


def test_attach_plugin():
    print("\nTesting attach_genesis_plugin helper")
    base = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 8))
    plugin = GenesisPlugin(hidden_size=16, output_size=5, vocab_size=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = base.to(device)

    # Force plugin on the opposite device (if possible) to test relocation
    other_device = torch.device("cpu")
    if device.type == "cpu" and torch.cuda.is_available():
        other_device = torch.device("cuda")
    plugin = plugin.to(other_device)

    handle = attach_genesis_plugin(base, plugin, layer_name="0")
    assert next(plugin.parameters()).device == device

    x = torch.randn(2, 10).to(device)
    _ = base(x)
    assert hasattr(base, "genesis_logits"), "Plugin logits not attached to base model"
    assert base.genesis_logits.shape == torch.Size([2, 5])
    handle.remove()
    print("attach_genesis_plugin helper works correctly")


def test_attach_plugin_with_grad():
    """Gradients should propagate from plugin logits when with_grad=True."""
    base = nn.Sequential(nn.Linear(4, 6))
    plugin = GenesisPlugin(hidden_size=6, output_size=3, vocab_size=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = base.to(device)

    handle = attach_genesis_plugin(base, plugin, layer_name="0", with_grad=True)
    assert next(plugin.parameters()).device == device

    optimizer = torch.optim.SGD(list(base.parameters()) + list(plugin.parameters()), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(2, 4).to(device)
    y = torch.randint(0, 3, (2,)).to(device)

    optimizer.zero_grad()
    _ = base(x)
    loss = criterion(base.genesis_logits, y)
    loss.backward()

    assert base[0].weight.grad is not None, "Base model did not receive gradients"
    assert plugin.decoder.weight.grad is not None, "Plugin parameters did not get gradients"

    optimizer.step()
    handle.remove()
    print("attach_genesis_plugin with_grad works correctly")


def test_replay_gradients_are_fresh():
    """Ensure replay updates do not reuse stale gradients."""
    input_dim = 10
    hidden_dim = 16
    output_dim = 20

    model = IntegratedLearningModule(input_dim, hidden_dim, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = StepCounterOptimizer(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x_batch = torch.randn(2, 3, input_dim).to(device)

    for i in range(10):
        y_batch = torch.randint(0, output_dim, (2,)).to(device)
        model.training_step(x_batch, y_batch, optimizer, criterion)

    assert optimizer.step_calls == 10
    assert optimizer.zero_calls == 10


def test_plugin_replay_gradients_are_fresh():
    """Same check for GenesisPlugin."""
    hidden_dim = 16
    output_dim = 20

    plugin = GenesisPlugin(hidden_dim, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plugin = plugin.to(device)

    optimizer = StepCounterOptimizer(plugin.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    hidden = torch.randn(2, hidden_dim).to(device)

    for i in range(10):
        y = torch.randint(0, output_dim, (2,)).to(device)
        plugin.training_step(hidden, y, optimizer, criterion)

    assert optimizer.step_calls == 10
    assert optimizer.zero_calls == 10


def test_apply_consolidation_penalty():
    """Penalty should be zero when no importance and positive otherwise."""
    model = IntegratedLearningModule(4, 6, 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # No importance -> zero penalty
    model.importance_scores.zero_()
    penalty_zero = model.apply_consolidation(lambda_reg=1.0)
    assert penalty_zero.item() == 0.0

    # Set importance on one unit and adjust bias to incur cost
    model.importance_scores[0] = 1.0
    model.anchor_bias.data[0] = 0.5
    penalty_nonzero = model.apply_consolidation(lambda_reg=1.0)
    assert penalty_nonzero.item() > 0.0


def test_consolidation_penalty_effect():
    """Large lambda_reg keeps anchor bias closer to its reference."""
    torch.manual_seed(0)
    model_low = IntegratedLearningModule(4, 6, 3)
    model_high = IntegratedLearningModule(4, 6, 3)
    model_high.load_state_dict(model_low.state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_low = model_low.to(device)
    model_high = model_high.to(device)

    for m in (model_low, model_high):
        m.importance_scores.fill_(1.0)
        m.anchor_bias.data.fill_(0.1)
        m.anchor_bias_ref.zero_()

    opt_low = torch.optim.SGD(model_low.parameters(), lr=0.1)
    opt_high = torch.optim.SGD(model_high.parameters(), lr=0.1)
    crit = nn.CrossEntropyLoss()

    x = torch.randn(2, 3, 4).to(device)
    y = torch.randint(0, 3, (2,)).to(device)

    model_low.training_step(x, y, opt_low, crit, lambda_reg=0.0)
    dist_low = model_low.anchor_bias.abs().sum().item()

    model_high.training_step(x, y, opt_high, crit, lambda_reg=5.0)
    dist_high = model_high.anchor_bias.abs().sum().item()

    assert dist_high < dist_low


def test_anchor_bias_ref_update_threshold():
    """Anchor ref should update only when gradients cross the threshold."""
    hidden_dim = 8
    output_dim = 3
    plugin = GenesisPlugin(hidden_dim, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plugin = plugin.to(device)
    opt = torch.optim.Adam(plugin.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()

    hidden = torch.ones(2, hidden_dim, device=device)
    target = torch.zeros(2, dtype=torch.long, device=device)

    # Zero decoder weights to keep gradients tiny -> no update
    plugin.decoder.weight.data.zero_()
    plugin.decoder.bias.data.zero_()
    ref_before = plugin.anchor_bias_ref.clone()
    plugin.training_step(hidden, target, opt, crit)
    assert torch.equal(ref_before, plugin.anchor_bias_ref)

    # Make first output weight huge so gradients spike on hidden[0]
    plugin.decoder.weight.data.zero_()
    plugin.decoder.weight.data[0, 0] = 100.0
    target.fill_(1)
    ref_before = plugin.anchor_bias_ref.clone()
    for _ in range(3):
        plugin.training_step(hidden, target, opt, crit)
        if plugin.importance_scores.sum() > 0:
            break
    assert plugin.importance_scores.sum() > 0
    assert not torch.equal(ref_before, plugin.anchor_bias_ref)


def test_replay_buffer_sampling_after_many_steps():
    """Sampling should still work after numerous training steps."""
    hidden_dim = 8
    output_dim = 4
    plugin = GenesisPlugin(hidden_dim, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plugin = plugin.to(device)

    opt = torch.optim.Adam(plugin.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()

    for _ in range(40):
        h = torch.randn(3, hidden_dim).to(device)
        y = torch.randint(0, output_dim, (3,)).to(device)
        plugin.training_step(h, y, opt, crit)

    h_sample, t_sample = plugin.replay_buffer.sample(batch_size=2, device=device)
    assert h_sample is not None and t_sample is not None
    assert h_sample.shape == torch.Size([2, hidden_dim])
    assert t_sample.shape == torch.Size([2])


def test_update_priority_affects_sampling():
    """Updating priorities should change sampling probabilities."""
    buf = SelfReplayBuffer(max_size=3)
    for i in range(3):
        buf.add(torch.tensor([float(i)]), torch.tensor(i), priority=1.0)

    torch.manual_seed(0)
    h_before, _ = buf.sample(batch_size=50)
    count_before = (h_before[:, 0] == 0.0).sum().item()

    buf.update_priority(0, 10.0)

    torch.manual_seed(0)
    h_after, _ = buf.sample(batch_size=50)
    count_after = (h_after[:, 0] == 0.0).sum().item()

    assert count_after > count_before, "Priority update did not affect sampling"


def test_update_priority_invalid_indices():
    """update_priority should raise IndexError for out-of-range indices."""
    buf = SelfReplayBuffer(max_size=2)
    for i in range(2):
        buf.add(torch.tensor([float(i)]), torch.tensor(i), priority=1.0)

    with pytest.raises(IndexError):
        buf.update_priority(5, 1.0)
    with pytest.raises(IndexError):
        buf.update_priority(-1, 1.0)
    with pytest.raises(IndexError):
        buf.update_priority([0, 2], [1.0, 2.0])


def test_sampling_with_zero_priorities():
    """Sampling should work even when all priorities are zero."""
    buf = SelfReplayBuffer(max_size=3)
    for i in range(3):
        buf.add(torch.tensor([float(i)]), torch.tensor(i), priority=0.0)

    h_sample, t_sample = buf.sample(batch_size=2)
    assert h_sample is not None and t_sample is not None
    assert h_sample.shape == torch.Size([2, 1])
    assert t_sample.shape == torch.Size([2])


def test_negative_priority_clamped():
    """Negative priorities should be clamped to zero on insertion."""
    buf = SelfReplayBuffer(max_size=1)
    buf.add(torch.tensor([1.0]), torch.tensor(0), priority=-5.0)

    assert buf.buffer[0][2] == 0.0

    h, t = buf.sample(batch_size=1)
    assert h is not None and t is not None


def test_sample_inconsistent_shapes(monkeypatch):
    """Sampling should raise an error when hidden shapes differ."""
    buf = SelfReplayBuffer(max_size=2)
    buf.add(torch.randn(2, 4), torch.tensor(0))
    buf.add(torch.randn(3, 4), torch.tensor(1))

    def fake_multinomial(probs, num, replacement=True):
        return torch.tensor([0, 1])

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    with pytest.raises(ValueError):
        buf.sample(batch_size=2)


def test_classifier_based_filtering():
    """EthicalGate with classifier should further reduce unsafe logits."""
    vocab = 5
    disallowed = [0]
    plugin = GenesisPlugin(
        hidden_size=4,
        output_size=vocab,
        vocab_size=vocab,
        disallowed_tokens=disallowed,
        gate_use_classifier=True,
        gate_classifier_scale=2.0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plugin = plugin.to(device)

    # Manually set classifier to strongly penalize token 2
    with torch.no_grad():
        for p in plugin.ethical_classifier.parameters():
            p.zero_()
        plugin.ethical_classifier[2].bias[:] = torch.tensor([5.0, 5.0, -5.0, 5.0, 5.0], device=device)

    logits = torch.zeros(1, vocab, device=device)
    filtered = plugin.gate.filter_logits_with_classifier(logits.clone(), plugin)

    assert filtered[0, 0] < -1e8  # mask still applied
    assert filtered[0, 2] < logits[0, 2] - 1.0  # penalized by classifier


def test_anchor_bias_clamped_after_many_steps_plugin():
    """Anchor bias should remain within [-bias_max, bias_max] after repeated training."""
    hidden_dim = 6
    output_dim = 5
    bias_max = 0.3
    plugin = GenesisPlugin(hidden_dim, output_dim, bias_max=bias_max, bias_decay=0.98)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plugin = plugin.to(device)
    opt = torch.optim.Adam(plugin.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()

    for _ in range(60):
        h = torch.randn(4, hidden_dim).to(device)
        y = torch.randint(0, output_dim, (4,)).to(device)
        plugin.training_step(h, y, opt, crit)

    assert torch.all(plugin.anchor_bias.abs() <= bias_max + 1e-6)


def test_anchor_bias_clamped_after_many_steps_module():
    """Same clamp check for IntegratedLearningModule."""
    input_dim = 4
    hidden_dim = 8
    output_dim = 6
    bias_max = 0.2
    model = IntegratedLearningModule(input_dim, hidden_dim, output_dim, bias_max=bias_max, bias_decay=0.98)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(3, 5, input_dim).to(device)

    for _ in range(60):
        y = torch.randint(0, output_dim, (3,)).to(device)
        model.training_step(x, y, opt, crit)

    assert torch.all(model.anchor_bias.abs() <= bias_max + 1e-6)


if __name__ == "__main__":
    test_genesis_module()
    test_genesis_plugin()
    test_attach_plugin()
    test_attach_plugin_with_grad()
    test_apply_consolidation_penalty()
    test_consolidation_penalty_effect()
    test_anchor_bias_ref_update_threshold()
    test_replay_buffer_sampling_after_many_steps()
    test_update_priority_affects_sampling()
    test_update_priority_invalid_indices()
    test_negative_priority_clamped()
    test_classifier_based_filtering()
    test_anchor_bias_clamped_after_many_steps_plugin()
    test_anchor_bias_clamped_after_many_steps_module()
