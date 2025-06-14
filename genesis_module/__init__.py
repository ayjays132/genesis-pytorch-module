import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfReplayBuffer:
    """Dynamic replay buffer storing internal representations (hidden states) and targets."""
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = []  # list of (hidden_repr, target) tuples

    def add(self, hidden, target):
        # Detach hidden state from computation graph and optionally move to CPU for storage
        hidden_detached = hidden.detach().cpu()
        target_detached = target.detach().cpu()
        # Add to buffer
        self.buffer.append((hidden_detached, target_detached))
        if len(self.buffer) > self.max_size:
            # Remove oldest entry to keep memory bounded
            self.buffer.pop(0)

    def sample(self, batch_size=1, device=None):
        """Sample a batch of stored (hidden, target) pairs."""
        if len(self.buffer) == 0:
            return None, None
        # Random sample with replacement or without depending on use-case
        indices = torch.randperm(len(self.buffer))[:batch_size]
        hiddens = []
        targets = []
        for idx in indices:
            h, t = self.buffer[idx]
            # Move to target device if specified
            if device:
                h = h.to(device)
                t = t.to(device)
            hiddens.append(h)
            targets.append(t)
        # Stack into tensors for batch processing
        # hidden shape might be [hidden_dim] or [seq_len, hidden_dim]; ensure consistent usage
        hiddens = torch.stack(hiddens, dim=0)
        targets = torch.stack(targets, dim=0)
        return hiddens, targets

class EthicalGate:
    """Ethical gating layer to filter/adjust logits according to allowed tokens."""
    def __init__(self, vocab_size, disallowed_tokens=None):
        self.vocab_size = vocab_size
        # Create a mask tensor for logits: 0 for disallowed tokens, 1 for allowed ones
        mask = torch.ones(vocab_size, dtype=torch.float32)
        if disallowed_tokens is not None:
            for tok in disallowed_tokens:
                if 0 <= tok < vocab_size:
                    mask[tok] = 0.0
        # Register the mask as a buffer (not a parameter, but moves with device).
        # We'll register the mask buffer once we have a nn.Module to attach to.
        # Here we'll just store it for later use in forward.
        self.registered = False
        self.mask = mask

    def register_to_module(self, module):
        """Register mask as buffer to a given module (so it moves to CUDA with module)."""
        module.register_buffer("ethical_mask", self.mask)
        self.registered = True

    def filter_logits(self, logits, module):
        """
        Adjust logits by masking disallowed tokens.
        logits: Tensor of shape [batch, vocab_size] (or [vocab_size] for 1D)
        module: the parent module containing the 'ethical_mask' buffer.
        """
        if not self.registered:
            # If not registered as buffer, assume mask is on same device as logits for simplicity
            mask = self.mask.to(logits.device)
        else:
            mask = module.ethical_mask  # use the registered buffer on the module
        # Set disallowed logits to a large negative value (approx -inf)
        # We add a very large negative number where mask is 0.
        # (mask==1 -> add 0, mask==0 -> add -1e9)
        neg_inf = -1e9
        # Ensure mask has same shape as logits (could add broadcasting for batch)
        if mask.dim() == 1 and logits.dim() == 2:
            mask_vec = mask.view(1, -1)  # shape [1, vocab_size]
        else:
            mask_vec = mask
        filtered_logits = logits + (mask_vec.to(logits.device) * 0.0 + (1 - mask_vec.to(logits.device)) * neg_inf)
        return filtered_logits

class IntegratedLearningModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size=None, disallowed_tokens=None):
        """
        input_size: dimension of input features (e.g. embedding size)
        hidden_size: dimension of hidden state in the core model
        output_size: dimension of model output (e.g. number of classes or vocab tokens)
        vocab_size: vocabulary size for ethical gating (if output is a distribution over vocab)
        disallowed_tokens: list of token indices to block via ethical gate
        """
        super(IntegratedLearningModule, self).__init__()
        # Core model: an LSTM encoder + linear decoder (for demonstration purposes)
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)
        # Replay buffer
        self.replay_buffer = SelfReplayBuffer(max_size=500)
        # Sticky amplifier persistence: anchor biases and importance matrix
        # Anchor bias for hidden state (1D parameter of length hidden_size)
        self.anchor_bias = nn.Parameter(torch.zeros(hidden_size))
        # Importance scores for weights (for simplicity, one score per hidden unit for now)
        # This can be extended to per-weight matrix; here we track importance per hidden unit
        self.register_buffer("importance_scores", torch.zeros(hidden_size))
        # Ethical gate
        self.gate = None
        if vocab_size:
            self.gate = EthicalGate(vocab_size, disallowed_tokens)
            # Register the ethical mask as buffer to this module so it moves with .to(device)
            self.register_buffer("ethical_mask", self.gate.mask)
            self.gate.registered = True  # mark that mask is registered
        # Novelty/abstraction metrics tracking
        self.register_buffer("novelty_score", torch.tensor(0.0))
        self.register_buffer("steps", torch.tensor(0))  # count training steps for scheduling

    def forward(self, x, hidden_in=None):
        """
        Forward pass.
        x: input tensor of shape [batch, seq_len, input_size]
        hidden_in: optional initial hidden state for LSTM (h0, c0)
        Returns: logits (filtered if ethical gate is on) and raw logits before filter.
        """
        batch_size = x.size(0)
        # LSTM encoding
        out, (h_n, c_n) = self.encoder(x, hidden_in)  # out: [batch, seq_len, hidden_size]
        # Take final hidden state of last time step for each sequence
        # (assuming we want one output per sequence, e.g., next-token prediction or classification)
        final_hidden = out[:, -1, :]  # shape [batch, hidden_size]
        # Apply sticky anchor bias to hidden state (amplifier persistence influence)
        final_hidden = final_hidden + self.anchor_bias  # broadcast add bias to each batch element
        # Decode to output logits
        logits = self.decoder(final_hidden)  # shape [batch, output_size]
        raw_logits = logits.clone()
        # Ethical gating: filter logits if gate is defined (assuming output are vocab probabilities)
        if self.gate:
            logits = self.gate.filter_logits(logits, module=self)  # uses the registered mask buffer
        return logits, raw_logits, final_hidden

    def training_step(self, x, target, optimizer, criterion):
        """
        Perform one training step: forward pass, loss computation, backward pass with amplifier, 
        persistence update, and optional replay.
        """
        self.train()
        optimizer.zero_grad()
        # Forward pass
        logits, raw_logits, final_hidden = self.forward(x)
        # Calculate main task loss
        loss_main = criterion(logits, target)
        # Compute novelty metric (e.g. current batch average loss as proxy for novelty)
        novelty = loss_main.detach()
        # Update novelty_score EMA (Exponential Moving Average)
        self.novelty_score = 0.9 * self.novelty_score + 0.1 * novelty
        # Backpropagate loss
        grad_hidden = torch.autograd.grad(loss_main, final_hidden, retain_graph=True)[0]
        # Sticky Learning Amplifier: detect high-gradient hidden units and amplify
        if grad_hidden is not None:
            # Compute gradient norm per hidden unit (across batch)
            grad_norm = grad_hidden.abs().mean(dim=0)  # mean absolute grad for each hidden unit
            # Determine which hidden units to amplify (gradient above threshold)
            threshold = grad_norm.mean() + 2 * grad_norm.std()  # e.g., 2 std above mean as threshold
            high_grad_mask = (grad_norm > threshold).float()  # 1 for important units
            if high_grad_mask.sum().item() > 0:
                # Amplify: Increase corresponding anchor_bias and importance_scores
                # Use current hidden activation as a reference for bias direction
                # (average across batch for each unit)
                avg_hidden = final_hidden.detach().mean(dim=0)
                # Update anchor bias: add a fraction of avg_hidden for units with high_grad
                amp_rate = 0.1  # small rate to update bias (hyperparameter)
                self.anchor_bias.data += amp_rate * high_grad_mask * avg_hidden
                # Update importance scores for persistence regularization
                self.importance_scores += high_grad_mask  # accumulate importance count (could also use grad_norm^2)
                # (We could also amplify gradient itself here or adjust learning rates dynamically if needed)
        # Optimizer step for main model parameters (with amplified adjustments already applied to gradients)
        # Note: anchor_bias is a parameter, so its gradient (if any) was computed; we manually modified it above.
        # We might want to zero its grad after manual update to avoid double counting, but since we updated .data, its grad is unchanged.
        # Simulate weight consolidation regularization: apply penalty for important weights deviating from reference.
        # For simplicity, suppose reference weights are initial weights (or we could store a snapshot). Not fully implemented here.
        # Clip gradients to maintain stable GradNorm (if any grad is too large)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        optimizer.step()
        # After weight update, one could apply an EWC-like regularization manually:
        # self.apply_consolidation()
        # (Not fully shown; would add loss term for (param - param_ref)^2 * importance_scores)
        # Add experience to replay buffer (pick a random sample from batch to store to limit size)
        idx = torch.randint(0, x.size(0), (1,)).item()
        # Store the final hidden state (with no grad) and corresponding target
        self.replay_buffer.add(final_hidden[idx], target[idx])
        self.steps += 1
        # Optionally perform a replay update periodically (e.g. every few steps)
        loss_replay = torch.tensor(0.0)
        if self.steps % 10 == 0:
            replay_h, replay_t = self.replay_buffer.sample(batch_size=1, device=x.device)
            if replay_h is not None:
                # Forward pass on replay hidden through decoder
                replay_logits = self.decoder(replay_h + self.anchor_bias)  # include anchor bias here as well
                # No gating on replay loss to ensure we reinforce raw mapping
                loss_replay = criterion(replay_logits, replay_t)
                loss_replay.backward()
                # Apply small weight updates for replay (learning on old sample)
                optimizer.step()  # second optimizer step for replay (could also accumulate gradients and do one step)
        # Total loss for reporting (main + any replay if applied)
        total_loss = loss_main.item() + loss_replay.item()
        return total_loss

    # (Optional) A method to apply consolidation regularization if we maintained reference weights
    def apply_consolidation(self, lambda_reg=1.0):
        # Pseudocode: for each param, if importance_scores for corresponding hidden unit is high, 
        # add gradient toward param_ref (not implemented fully due to simplicity of importance in this example).
        # This would be integrated in loss or after backward.
        pass




class GenesisPlugin(nn.Module):
    """Attachable module providing GENESIS functionality for any base model.

    Parameters
    ----------
    hidden_size : int
        Dimension of the hidden representation passed from the base model.
    output_size : int
        Dimension of the output logits produced by this plugin.
    vocab_size : int, optional
        Vocabulary size for ethical gating. If provided, logits will be filtered
        to suppress disallowed tokens.
    disallowed_tokens : list of int, optional
        Token indices that should be suppressed by the ethical gate.
    replay_size : int, optional
        Maximum number of items in the replay buffer.
    """

    def __init__(self, hidden_size, output_size, vocab_size=None,
                 disallowed_tokens=None, replay_size=500):
        super().__init__()
        self.hidden_size = hidden_size
        self.decoder = nn.Linear(hidden_size, output_size)
        self.replay_buffer = SelfReplayBuffer(max_size=replay_size)
        self.anchor_bias = nn.Parameter(torch.zeros(hidden_size))
        self.register_buffer("importance_scores", torch.zeros(hidden_size))
        self.gate = None
        if vocab_size:
            self.gate = EthicalGate(vocab_size, disallowed_tokens)
            self.register_buffer("ethical_mask", self.gate.mask)
            self.gate.registered = True
        self.register_buffer("novelty_score", torch.tensor(0.0))
        self.register_buffer("steps", torch.tensor(0))

    def forward(self, hidden):
        """Apply GENESIS biasing and compute logits."""
        final_hidden = hidden + self.anchor_bias
        logits = self.decoder(final_hidden)
        raw_logits = logits.clone()
        if self.gate:
            logits = self.gate.filter_logits(logits, module=self)
        return logits, raw_logits, final_hidden

    def training_step(self, hidden, target, optimizer, criterion):
        """Perform a training step using provided hidden states."""
        self.train()
        optimizer.zero_grad()
        logits, raw_logits, final_hidden = self.forward(hidden)
        loss_main = criterion(logits, target)
        novelty = loss_main.detach()
        self.novelty_score = 0.9 * self.novelty_score + 0.1 * novelty
        grad_hidden = torch.autograd.grad(loss_main, final_hidden, retain_graph=True)[0]
        if grad_hidden is not None:
            grad_norm = grad_hidden.abs().mean(dim=0)
            threshold = grad_norm.mean() + 2 * grad_norm.std()
            high_grad_mask = (grad_norm > threshold).float()
            if high_grad_mask.sum().item() > 0:
                avg_hidden = final_hidden.detach().mean(dim=0)
                amp_rate = 0.1
                self.anchor_bias.data += amp_rate * high_grad_mask * avg_hidden
                self.importance_scores += high_grad_mask
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        optimizer.step()
        idx = torch.randint(0, hidden.size(0), (1,)).item()
        self.replay_buffer.add(final_hidden[idx], target[idx])
        self.steps += 1
        loss_replay = torch.tensor(0.0)
        if self.steps % 10 == 0:
            replay_h, replay_t = self.replay_buffer.sample(batch_size=1, device=hidden.device)
            if replay_h is not None:
                replay_logits = self.decoder(replay_h + self.anchor_bias)
                loss_replay = criterion(replay_logits, replay_t)
                loss_replay.backward()
                optimizer.step()
        total_loss = loss_main.item() + loss_replay.item()
        return total_loss


from .integration import attach_genesis_plugin

