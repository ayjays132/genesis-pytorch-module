import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfReplayBuffer:
    """Dynamic replay buffer for hidden representations and targets.

    Parameters
    ----------
    max_size : int, optional
        Maximum number of items to keep.
    dtype : torch.dtype, optional
        Data type used to store the hidden representations. ``float16`` reduces
        memory footprint while having negligible impact on accuracy for
        normalized activations.
    """

    def __init__(self, max_size=1000, dtype=torch.float16):
        self.max_size = max_size
        self.dtype = dtype
        # list of tuples: (hidden_repr, target, priority)
        self.buffer = []

    def add(self, hidden, target, priority=1.0):
        """Store a hidden state and target with an associated priority."""
        hidden_detached = hidden.detach().to(self.dtype).cpu()
        target_detached = target.detach().cpu()
        self.buffer.append((hidden_detached, target_detached, float(priority)))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size=1, device=None):
        """Sample a batch of stored (hidden, target) pairs."""
        if len(self.buffer) == 0:
            return None, None
        priorities = torch.tensor([p for (_, _, p) in self.buffer], dtype=torch.float)
        probs = priorities / priorities.sum()
        indices = torch.multinomial(probs, batch_size, replacement=True)
        hiddens, targets = [], []
        for idx in indices:
            h, t, _ = self.buffer[int(idx)]
            if device:
                h = h.to(device=device, dtype=torch.float32)
                t = t.to(device)
            else:
                h = h.to(dtype=torch.float32)
            hiddens.append(h)
            targets.append(t)
        # Stack into tensors for batch processing
        # hidden shape might be [hidden_dim] or [seq_len, hidden_dim]; ensure consistent usage
        hiddens = torch.stack(hiddens, dim=0)
        targets = torch.stack(targets, dim=0)
        return hiddens, targets

    def update_priority(self, index, value):
        """Update priority of stored item(s).

        Parameters
        ----------
        index : int or sequence of ints
            Index or indices of items in the buffer to update.
        value : float or sequence of floats
            New priority value(s). If ``value`` is a single float and ``index``
            is a sequence, the same value is applied to all specified indices.
        """
        if isinstance(index, (list, tuple, torch.Tensor)):
            if isinstance(value, (list, tuple, torch.Tensor)):
                if len(index) != len(value):
                    raise ValueError("index and value must have the same length")
                pairs = zip(index, value)
            else:
                pairs = [(i, value) for i in index]
            for idx, val in pairs:
                idx = int(idx)
                if 0 <= idx < len(self.buffer):
                    h, t, _ = self.buffer[idx]
                    self.buffer[idx] = (h, t, float(val))
        else:
            idx = int(index)
            if 0 <= idx < len(self.buffer):
                h, t, _ = self.buffer[idx]
                self.buffer[idx] = (h, t, float(value))

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
        # Persistent reference of anchor bias for consolidation
        self.register_buffer("anchor_bias_ref", torch.zeros(hidden_size))
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

    def update_learning_rate(self, optimizer, base_lr=1e-3, min_lr=1e-5, max_lr=1e-2):
        """Dynamically adjust learning rate based on novelty_score."""
        novelty_val = float(self.novelty_score)
        progress = 1.0 - torch.tanh(torch.tensor(novelty_val)).item()
        lr = base_lr + (max_lr - base_lr) * progress
        lr = max(min_lr, min(max_lr, lr))
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    def training_step(self, x, target, optimizer, criterion):
        """
        Perform one training step: forward pass, loss computation, backward pass with amplifier, 
        persistence update, and optional replay.
        """
        self.train()
        optimizer.zero_grad()
        # Forward pass
        logits, raw_logits, final_hidden = self.forward(x)
        # Retain gradient on final_hidden so the amplifier can inspect it
        final_hidden.retain_grad()
        # Calculate main task loss
        loss_main = criterion(logits, target)
        # Compute novelty metric (e.g. current batch average loss as proxy for novelty)
        novelty = loss_main.detach()
        # Update novelty_score EMA (Exponential Moving Average)
        self.novelty_score = 0.9 * self.novelty_score + 0.1 * novelty
        # Adjust learning rate according to novelty
        self.update_learning_rate(optimizer)
        # Backpropagate loss
        loss_main.backward(retain_graph=True)
        grad_hidden = final_hidden.grad
        # Sticky Learning Amplifier: detect high-gradient hidden units and amplify
        if grad_hidden is not None:
            # Compute gradient norm per hidden unit (across batch)
            grad_norm = grad_hidden.abs().mean(dim=0)  # mean absolute grad for each hidden unit
            # Determine which hidden units to amplify (gradient above threshold)
            threshold = grad_norm.mean() + 2 * grad_norm.std()  # e.g., 2 std above mean as threshold
            high_grad_mask = (grad_norm > threshold).float()  # 1 for important units
            if high_grad_mask.sum().item() > 0:
                # Amplify: increase anchor_bias for salient units
                avg_hidden = final_hidden.detach().mean(dim=0)
                amp_rate = 0.1
                self.anchor_bias.data += amp_rate * high_grad_mask * avg_hidden
                self.importance_scores += high_grad_mask
                # (We could also amplify gradient itself here or adjust learning rates dynamically if needed)
        # Consolidation regularization to preserve important anchor_bias values
        if self.importance_scores.sum() > 0:
            reg_loss = torch.sum(self.importance_scores * (self.anchor_bias - self.anchor_bias_ref) ** 2)
            reg_loss = 0.1 * reg_loss
            reg_loss.backward(retain_graph=True)
        # Add experience to replay buffer (pick a random sample from batch to store to limit size)
        idx = torch.randint(0, x.size(0), (1,)).item()
        priority = float(loss_main.detach())
        self.replay_buffer.add(final_hidden[idx], target[idx], priority=priority)
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
                # Accumulate replay gradients with main gradients
                loss_replay.backward()
        # Clip gradients to maintain stable GradNorm (if any grad is too large)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        optimizer.step()
        self.anchor_bias_ref = self.anchor_bias.detach().clone()
        # Total loss for reporting (main + any replay if applied)
        total_loss = loss_main.item() + loss_replay.item()
        return total_loss

    # (Optional) A method to apply consolidation regularization if we maintained reference weights
    def apply_consolidation(self, lambda_reg=0.1):
        """Apply consolidation penalty to anchor_bias based on importance scores."""
        if self.importance_scores.sum() == 0:
            return torch.tensor(0.0, device=self.anchor_bias.device)
        penalty = lambda_reg * torch.sum(self.importance_scores * (self.anchor_bias - self.anchor_bias_ref) ** 2)
        return penalty




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
        self.register_buffer("anchor_bias_ref", torch.zeros(hidden_size))
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

    def update_learning_rate(self, optimizer, base_lr=1e-3, min_lr=1e-5, max_lr=1e-2):
        """Adjust optimizer learning rate using the current novelty_score."""
        novelty_val = float(self.novelty_score)
        progress = 1.0 - torch.tanh(torch.tensor(novelty_val)).item()
        lr = base_lr + (max_lr - base_lr) * progress
        lr = max(min_lr, min(max_lr, lr))
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    def training_step(self, hidden, target, optimizer, criterion):
        """Perform a training step using provided hidden states."""
        self.train()
        optimizer.zero_grad()
        logits, raw_logits, final_hidden = self.forward(hidden)
        # Retain gradient on final_hidden so the amplifier can access it
        final_hidden.retain_grad()
        loss_main = criterion(logits, target)
        novelty = loss_main.detach()
        self.novelty_score = 0.9 * self.novelty_score + 0.1 * novelty
        self.update_learning_rate(optimizer)
        loss_main.backward(retain_graph=True)
        grad_hidden = final_hidden.grad
        if grad_hidden is not None:
            grad_norm = grad_hidden.abs().mean(dim=0)
            threshold = grad_norm.mean() + 2 * grad_norm.std()
            high_grad_mask = (grad_norm > threshold).float()
            if high_grad_mask.sum().item() > 0:
                avg_hidden = final_hidden.detach().mean(dim=0)
                amp_rate = 0.1
                self.anchor_bias.data += amp_rate * high_grad_mask * avg_hidden
                self.importance_scores += high_grad_mask
        if self.importance_scores.sum() > 0:
            reg_loss = 0.1 * torch.sum(self.importance_scores * (self.anchor_bias - self.anchor_bias_ref) ** 2)
            reg_loss.backward(retain_graph=True)
        idx = torch.randint(0, hidden.size(0), (1,)).item()
        priority = float(loss_main.detach())
        self.replay_buffer.add(final_hidden[idx], target[idx], priority=priority)
        self.steps += 1
        loss_replay = torch.tensor(0.0)
        if self.steps % 10 == 0:
            replay_h, replay_t = self.replay_buffer.sample(batch_size=1, device=hidden.device)
            if replay_h is not None:
                replay_logits = self.decoder(replay_h + self.anchor_bias)
                loss_replay = criterion(replay_logits, replay_t)
                # Accumulate replay gradients with main gradients
                loss_replay.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        optimizer.step()
        self.anchor_bias_ref = self.anchor_bias.detach().clone()
        total_loss = loss_main.item() + loss_replay.item()
        return total_loss

    def apply_consolidation(self, lambda_reg=0.1):
        """Return consolidation penalty based on importance scores."""
        if self.importance_scores.sum() == 0:
            return torch.tensor(0.0, device=self.anchor_bias.device)
        penalty = lambda_reg * torch.sum(self.importance_scores * (self.anchor_bias - self.anchor_bias_ref) ** 2)
        return penalty


from .integration import attach_genesis_plugin

