import torch
import torch.nn as nn

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
        total = priorities.sum()
        if total == 0:
            probs = torch.ones_like(priorities) / len(priorities)
        else:
            probs = priorities / total
        indices = torch.multinomial(probs, batch_size, replacement=True)
        hiddens, targets = [], []
        shapes = []
        for idx in indices:
            h, t, _ = self.buffer[int(idx)]
            if device:
                h = h.to(device=device, dtype=torch.float32)
                t = t.to(device)
            else:
                h = h.to(dtype=torch.float32)
            hiddens.append(h)
            targets.append(t)
            shapes.append(tuple(h.shape))
        # Stack into tensors for batch processing
        # hidden shape might be [hidden_dim] or [seq_len, hidden_dim]; ensure consistent usage
        if len(set(shapes)) > 1:
            raise ValueError(f"Inconsistent hidden state shapes: {shapes}")
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

        Raises
        ------
        IndexError
            If any provided index is outside ``[0, len(buffer))``.
        """
        if isinstance(index, (list, tuple, torch.Tensor)):
            indices = [int(i) for i in index]
            if any(i < 0 or i >= len(self.buffer) for i in indices):
                raise IndexError("index out of range")
            if isinstance(value, (list, tuple, torch.Tensor)):
                if len(indices) != len(value):
                    raise ValueError("index and value must have the same length")
                values = [float(v) for v in value]
            else:
                values = [float(value)] * len(indices)
            for idx, val in zip(indices, values):
                h, t, _ = self.buffer[idx]
                self.buffer[idx] = (h, t, val)
        else:
            idx = int(index)
            if idx < 0 or idx >= len(self.buffer):
                raise IndexError("index out of range")
            h, t, _ = self.buffer[idx]
            self.buffer[idx] = (h, t, float(value))

class EthicalGate:
    """Ethical gating layer to filter/adjust logits according to allowed tokens.

    Parameters
    ----------
    vocab_size : int
        Size of the output vocabulary.
    disallowed_tokens : list[int], optional
        Tokens that should always be masked out.
    use_classifier : bool, optional
        If ``True`` an internal MLP classifier scores tokens for safety and can
        further reduce unsafe token probabilities.
    classifier_hidden : int, optional
        Hidden dimension of the MLP classifier.
    classifier_scale : float, optional
        Multiplicative scale applied to the classifier score when reducing
        logits.
    """

    def __init__(self, vocab_size, disallowed_tokens=None, *, use_classifier=False,
                 classifier_hidden=32, classifier_scale=5.0):
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
        self.use_classifier = use_classifier
        self.classifier_scale = classifier_scale
        if use_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(vocab_size, classifier_hidden),
                nn.ReLU(),
                nn.Linear(classifier_hidden, vocab_size),
                nn.Sigmoid(),
            )

    def register_to_module(self, module):
        """Register mask as buffer to a given module (so it moves to CUDA with module)."""
        module.register_buffer("ethical_mask", self.mask)
        if self.use_classifier:
            module.add_module("ethical_classifier", self.classifier)
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

    def filter_logits_with_classifier(self, logits, module):
        """Filter logits using the static mask and optional classifier."""
        logits = self.filter_logits(logits, module)
        if not self.use_classifier:
            return logits
        safety_scores = self.classifier(logits.detach())  # [batch, vocab]
        logits = logits - self.classifier_scale * (1.0 - safety_scores.to(logits.device))
        return logits

class GenesisCore(nn.Module):
    """Mixin providing replay, amplification, and gating utilities."""

    def __init__(self, hidden_size, output_size, vocab_size=None, disallowed_tokens=None,
                 replay_size=500, bias_decay: float = 0.999, bias_max: float = 5.0,
                 gate_use_classifier: bool = False, gate_classifier_hidden: int = 32,
                 gate_classifier_scale: float = 5.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.decoder = nn.Linear(hidden_size, output_size)
        self.replay_buffer = SelfReplayBuffer(max_size=replay_size)
        self.anchor_bias = nn.Parameter(torch.zeros(hidden_size))
        self.bias_decay = bias_decay
        self.bias_max = bias_max
        self.register_buffer("anchor_bias_ref", torch.zeros(hidden_size))
        self.register_buffer("importance_scores", torch.zeros(hidden_size))
        self.gate = None
        if vocab_size:
            self.gate = EthicalGate(
                vocab_size,
                disallowed_tokens,
                use_classifier=gate_use_classifier,
                classifier_hidden=gate_classifier_hidden,
                classifier_scale=gate_classifier_scale,
            )
            self.gate.register_to_module(self)
        self.register_buffer("novelty_score", torch.tensor(0.0))
        self.register_buffer("steps", torch.tensor(0))

    def core_forward(self, hidden):
        final_hidden = hidden + self.anchor_bias
        logits = self.decoder(final_hidden)
        raw_logits = logits.clone()
        if self.gate:
            if getattr(self.gate, "use_classifier", False):
                logits = self.gate.filter_logits_with_classifier(logits, module=self)
            else:
                logits = self.gate.filter_logits(logits, module=self)
        return logits, raw_logits, final_hidden

    def update_learning_rate(self, optimizer, base_lr=1e-3, min_lr=1e-5, max_lr=1e-2):
        novelty_val = float(self.novelty_score)
        progress = 1.0 - torch.tanh(torch.tensor(novelty_val)).item()
        lr = base_lr + (max_lr - base_lr) * progress
        lr = max(min_lr, min(max_lr, lr))
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _shared_training_logic(self, final_hidden, logits, target, optimizer, criterion):
        final_hidden.retain_grad()
        loss_main = criterion(logits, target)
        novelty = loss_main.detach()
        self.novelty_score = 0.9 * self.novelty_score + 0.1 * novelty
        self.update_learning_rate(optimizer)
        loss_main.backward(retain_graph=True)
        grad_hidden = final_hidden.grad
        amplifier_triggered = False
        if grad_hidden is not None:
            grad_norm = grad_hidden.abs().mean(dim=0)
            threshold = grad_norm.mean() + 2 * grad_norm.std()
            high_grad_mask = (grad_norm > threshold).float()
            if high_grad_mask.sum().item() > 0:
                amplifier_triggered = True
                avg_hidden = final_hidden.detach().mean(dim=0)
                amp_rate = 0.1
                self.anchor_bias.data += amp_rate * high_grad_mask * avg_hidden
                self.importance_scores += high_grad_mask
        if self.importance_scores.sum() > 0:
            reg_loss = 0.1 * torch.sum(self.importance_scores * (self.anchor_bias - self.anchor_bias_ref) ** 2)
            reg_loss.backward(retain_graph=True)
        idx = torch.randint(0, final_hidden.size(0), (1,)).item()
        priority = float(loss_main.detach())
        self.replay_buffer.add(final_hidden[idx], target[idx], priority=priority)
        self.steps += 1
        loss_replay = torch.tensor(0.0)
        if self.steps % 10 == 0:
            replay_h, replay_t = self.replay_buffer.sample(batch_size=1, device=final_hidden.device)
            if replay_h is not None:
                replay_logits = self.decoder(replay_h + self.anchor_bias)
                loss_replay = criterion(replay_logits, replay_t)
                loss_replay.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            torch.clamp_(self.anchor_bias, -self.bias_max, self.bias_max)
        if amplifier_triggered:
            self.anchor_bias_ref = self.anchor_bias.detach().clone()
        total_loss = loss_main.item() + loss_replay.item()
        return total_loss

    def apply_consolidation(self, lambda_reg=0.1):
        if self.importance_scores.sum() == 0:
            return torch.tensor(0.0, device=self.anchor_bias.device)
        penalty = lambda_reg * torch.sum(self.importance_scores * (self.anchor_bias - self.anchor_bias_ref) ** 2)
        return penalty
class IntegratedLearningModule(GenesisCore):
    def __init__(self, input_size, hidden_size, output_size, vocab_size=None, disallowed_tokens=None,
                 bias_decay: float = 0.999, bias_max: float = 5.0,
                 gate_use_classifier: bool = False, gate_classifier_hidden: int = 32,
                 gate_classifier_scale: float = 5.0):
        """
        input_size: dimension of input features (e.g. embedding size)
        hidden_size: dimension of hidden state in the core model
        output_size: dimension of model output (e.g. number of classes or vocab tokens)
        vocab_size: vocabulary size for ethical gating (if output is a distribution over vocab)
        disallowed_tokens: list of token indices to block via ethical gate
        gate_use_classifier: enable additional classifier-based filtering
        gate_classifier_hidden: hidden dimension for classifier if used
        gate_classifier_scale: scale factor for classifier penalty
        """
        super().__init__(
            hidden_size, output_size,
            vocab_size=vocab_size,
            disallowed_tokens=disallowed_tokens,
            bias_decay=bias_decay,
            bias_max=bias_max,
            gate_use_classifier=gate_use_classifier,
            gate_classifier_hidden=gate_classifier_hidden,
            gate_classifier_scale=gate_classifier_scale,
        )
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden_in=None):
        """Forward pass returning logits and hidden states."""
        out, _ = self.encoder(x, hidden_in)
        final_hidden = out[:, -1, :]
        return self.core_forward(final_hidden)

    def training_step(self, x, target, optimizer, criterion):
        self.train()
        optimizer.zero_grad()
        with torch.no_grad():
            self.anchor_bias.mul_(self.bias_decay)
            torch.clamp_(self.anchor_bias, -self.bias_max, self.bias_max)
        logits, raw_logits, final_hidden = self.forward(x)
        return self._shared_training_logic(final_hidden, logits, target, optimizer, criterion)





class GenesisPlugin(GenesisCore):
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
                 disallowed_tokens=None, replay_size=500,
                 bias_decay: float = 0.999, bias_max: float = 5.0,
                 gate_use_classifier: bool = False, gate_classifier_hidden: int = 32,
                 gate_classifier_scale: float = 5.0):
        super().__init__(
            hidden_size, output_size,
            vocab_size=vocab_size,
            disallowed_tokens=disallowed_tokens,
            replay_size=replay_size,
            bias_decay=bias_decay,
            bias_max=bias_max,
            gate_use_classifier=gate_use_classifier,
            gate_classifier_hidden=gate_classifier_hidden,
            gate_classifier_scale=gate_classifier_scale,
        )
    def forward(self, hidden):
        return self.core_forward(hidden)

    def training_step(self, hidden, target, optimizer, criterion):
        self.train()
        optimizer.zero_grad()
        with torch.no_grad():
            self.anchor_bias.mul_(self.bias_decay)
            torch.clamp_(self.anchor_bias, -self.bias_max, self.bias_max)
        logits, raw_logits, final_hidden = self.forward(hidden)
        return self._shared_training_logic(final_hidden, logits, target, optimizer, criterion)
from .integration import attach_genesis_plugin

__all__ = [
    "SelfReplayBuffer",
    "EthicalGate",
    "GenesisCore",
    "GenesisPlugin",
    "IntegratedLearningModule",
    "attach_genesis_plugin",
]

