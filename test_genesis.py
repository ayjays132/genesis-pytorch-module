import torch
import torch.nn as nn
from genesis_module import IntegratedLearningModule, SelfReplayBuffer, EthicalGate

def test_genesis_module():
    print("Starting GENESIS module test...")

    # Test 1: Initialization and basic forward pass
    print("\nTest 1: Initialization and basic forward pass")
    input_dim = 50
    hidden_dim = 128
    output_dim = 100
    vocab_size = output_dim
    disallowed = [0, 1, 2]

    model = IntegratedLearningModule(input_dim, hidden_dim, output_dim, vocab_size=vocab_size, disallowed_tokens=disallowed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model initialized and moved to {device}.")

    x_batch = torch.randn(4, 10, input_dim).to(device) # Batch size 4, seq_len 10
    logits, raw_logits, final_hidden = model(x_batch)

    assert logits.shape == torch.Size([4, output_dim]), f"Expected logits shape [4, {output_dim}], got {logits.shape}"
    assert raw_logits.shape == torch.Size([4, output_dim]), f"Expected raw_logits shape [4, {output_dim}], got {raw_logits.shape}"
    assert final_hidden.shape == torch.Size([4, hidden_dim]), f"Expected final_hidden shape [4, {hidden_dim}], got {final_hidden.shape}"
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
    initial_importance_scores = model.importance_scores.clone().detach()
    initial_novelty_score = model.novelty_score.clone().detach()

    # Run a few training steps to observe changes
    for i in range(5):
        y_batch = torch.randint(0, output_dim, (4,)).to(device)
        loss_val = model.training_step(x_batch, y_batch, optim, crit)
        print(f"  Step {i+1}, Loss: {loss_val:.4f}, Novelty Score: {model.novelty_score.item():.4f}")

    # Check if anchor_bias and importance_scores have changed (indicating amplifier activity)
    assert not torch.equal(initial_anchor_bias, model.anchor_bias), "Anchor bias did not change after training steps."
    assert not torch.equal(initial_importance_scores, model.importance_scores), "Importance scores did not change after training steps."
    assert not torch.equal(initial_novelty_score, model.novelty_score), "Novelty score did not change after training steps."
    print("Training steps successful: Anchor bias, importance scores, and novelty score updated.")

    # Test 4: Replay buffer functionality
    print("\nTest 4: Replay buffer functionality")
    # After several steps, the buffer should have some items
    assert len(model.replay_buffer.buffer) > 0, "Replay buffer is empty."
    print(f"Replay buffer contains {len(model.replay_buffer.buffer)} items.")

    # Sample from buffer and check shapes
    h_replay, t_replay = model.replay_buffer.sample(batch_size=1, device=device)
    assert h_replay is not None and t_replay is not None, "Failed to sample from replay buffer."
    assert h_replay.shape == torch.Size([1, hidden_dim]), f"Expected replay hidden shape [1, {hidden_dim}], got {h_replay.shape}"
    assert t_replay.shape == torch.Size([1]), f"Expected replay target shape [1], got {t_replay.shape}"
    print("Replay buffer sampling successful.")

    print("\nAll GENESIS module tests passed successfully!")

if __name__ == "__main__":
    test_genesis_module()


