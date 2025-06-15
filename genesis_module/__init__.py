import torch
import torch.nn as nn
from collections import deque


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

    Notes
    -----
    The ``priority`` value associated with each entry is clamped to be
    non-negative when added to the buffer. Any negative value will be stored
    as ``0.0``.
    """

    def __init__(self, max_size=1000, dtype=torch.float16, device=None):
        self.max_size = max_size
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        # deque of tuples: (hidden_repr, target, priority)
        self.buffer = deque(maxlen=self.max_size)

    def add(self, hidden, target, priority=1.0):
        """Store a hidden state and target with an associated priority."""
        hidden_detached = hidden.detach().to(device=self.device, dtype=self.dtype)
        target_detached = target.detach().to(self.device)
        priority_clamped = max(0.0, float(priority))
        # deque automatically discards oldest items when maxlen is reached
        self.buffer.append((hidden_detached, target_detached, priority_clamped))

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
        target_device = torch.device(device) if device is not None else self.device
        for idx in indices:
            h, t, _ = self.buffer[int(idx)]
            if device is not None and target_device != self.device:
                h = h.to(device=target_device, dtype=torch.float32)
                t = t.to(target_device)
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
                # ensure priorities remain non-negative just like in ``add``
                clamped = max(0.0, float(val))
                self.buffer[idx] = (h, t, clamped)
        else:
            idx = int(index)
            if idx < 0 or idx >= len(self.buffer):
                raise IndexError("index out of range")
            h, t, _ = self.buffer[idx]
            # Match ``add`` by clamping negative priorities to zero
            clamped = max(0.0, float(value))
            self.buffer[idx] = (h, t, clamped)


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

    def __init__(
        self, vocab_size, disallowed_tokens=None, *, use_classifier=False, classifier_hidden=32, classifier_scale=5.0
    ):
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
        """Register mask and optional classifier as submodules of ``module``.

        If ``module`` already defines an attribute called ``ethical_classifier``
        and this gate uses a classifier, a unique name ``ethical_classifier_X``
        is generated to avoid collisions where ``X`` is an incrementing integer
        starting at 1.
        """
        module.register_buffer("ethical_mask", self.mask)
        if self.use_classifier:
            name = "ethical_classifier"
            if hasattr(module, name):
                # Find a unique suffix
                idx = 1
                new_name = f"{name}_{idx}"
                while hasattr(module, new_name):
                    idx += 1
                    new_name = f"{name}_{idx}"
                name = new_name
            module.add_module(name, self.classifier)
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
    """Mixin providing replay, amplification, and gating utilities.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the internal hidden representation.
    output_size : int
        Dimensionality of the output logits.
    vocab_size : int, optional
        Vocabulary size for ethical gating.
    disallowed_tokens : list of int, optional
        Token indices to suppress via the ethical gate.
    replay_size : int, optional
        Maximum number of items stored in the replay buffer.
    bias_decay : float, optional
        Exponential decay applied to ``anchor_bias`` each step.
    bias_max : float, optional
        Clamp value for ``anchor_bias``.
    gate_use_classifier : bool, optional
        Enable classifier-based gating.
    gate_classifier_hidden : int, optional
        Hidden dimension for the gating classifier.
    gate_classifier_scale : float, optional
        Scale factor for classifier penalty.
    amp_rate : float, optional
        Scale factor for sticky learning amplifier updates.
    amp_threshold : float, optional
        Standard deviation multiplier for amplifier trigger.
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        vocab_size=None,
        disallowed_tokens=None,
        replay_size=500,
        bias_decay: float = 0.999,
        bias_max: float = 5.0,
        gate_use_classifier: bool = False,
        gate_classifier_hidden: int = 32,
        gate_classifier_scale: float = 5.0,
        amp_rate: float = 0.1,
        amp_threshold: float = 2.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.decoder = nn.Linear(hidden_size, output_size)
        self.replay_buffer = SelfReplayBuffer(max_size=replay_size)
        self.anchor_bias = nn.Parameter(torch.zeros(hidden_size))
        self.bias_decay = bias_decay
        self.bias_max = bias_max
        self.amp_rate = amp_rate
        self.amp_threshold = amp_threshold
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

    def _shared_training_logic(self, final_hidden, logits, target, optimizer, criterion, *, lambda_reg: float = 0.0):
        final_hidden.retain_grad()
        loss_main = criterion(logits, target)
        novelty = loss_main.detach()
        self.novelty_score = 0.9 * self.novelty_score + 0.1 * novelty
        self.update_learning_rate(optimizer)
        penalty = self.apply_consolidation(lambda_reg=lambda_reg)
        loss_total = loss_main + penalty
        loss_total.backward(retain_graph=True)
        grad_hidden = final_hidden.grad
        amplifier_triggered = False
        if grad_hidden is not None:
            grad_norm = grad_hidden.abs().mean(dim=0)
            threshold = grad_norm.mean() + self.amp_threshold * grad_norm.std()
            high_grad_mask = (grad_norm > threshold).float()
            if high_grad_mask.sum().item() > 0:
                amplifier_triggered = True
                avg_hidden = final_hidden.detach().mean(dim=0)
                self.anchor_bias.data += self.amp_rate * high_grad_mask * avg_hidden
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
        total_loss = loss_main.item() + penalty.item() + loss_replay.item()
        return total_loss

    def apply_consolidation(self, lambda_reg=0.1):
        if self.importance_scores.sum() == 0:
            return torch.tensor(0.0, device=self.anchor_bias.device)
        penalty = lambda_reg * torch.sum(self.importance_scores * (self.anchor_bias - self.anchor_bias_ref) ** 2)
        return penalty


class IntegratedLearningModule(GenesisCore):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        vocab_size=None,
        disallowed_tokens=None,
        bias_decay: float = 0.999,
        bias_max: float = 5.0,
        gate_use_classifier: bool = False,
        gate_classifier_hidden: int = 32,
        gate_classifier_scale: float = 5.0,
        amp_rate: float = 0.1,
        amp_threshold: float = 2.0,
    ):
        """
        input_size: dimension of input features (e.g. embedding size)
        hidden_size: dimension of hidden state in the core model
        output_size: dimension of model output (e.g. number of classes or vocab tokens)
        vocab_size: vocabulary size for ethical gating (if output is a distribution over vocab)
        disallowed_tokens: list of token indices to block via ethical gate
        gate_use_classifier: enable additional classifier-based filtering
        gate_classifier_hidden: hidden dimension for classifier if used
        gate_classifier_scale: scale factor for classifier penalty
        amp_rate: scale factor for sticky learning amplifier updates
        amp_threshold: standard deviation multiplier for amplifier trigger
        """
        super().__init__(
            hidden_size,
            output_size,
            vocab_size=vocab_size,
            disallowed_tokens=disallowed_tokens,
            bias_decay=bias_decay,
            bias_max=bias_max,
            gate_use_classifier=gate_use_classifier,
            gate_classifier_hidden=gate_classifier_hidden,
            gate_classifier_scale=gate_classifier_scale,
            amp_rate=amp_rate,
            amp_threshold=amp_threshold,
        )
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden_in=None):
        """Forward pass returning logits and hidden states."""
        out, _ = self.encoder(x, hidden_in)
        final_hidden = out[:, -1, :]
        return self.core_forward(final_hidden)

    def training_step(self, x, target, optimizer, criterion, *, lambda_reg: float = 0.0):
        self.train()
        optimizer.zero_grad()
        with torch.no_grad():
            self.anchor_bias.mul_(self.bias_decay)
            torch.clamp_(self.anchor_bias, -self.bias_max, self.bias_max)
        logits, raw_logits, final_hidden = self.forward(x)
        return self._shared_training_logic(final_hidden, logits, target, optimizer, criterion, lambda_reg=lambda_reg)


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
    amp_rate : float, optional
        Scale factor for sticky learning amplifier updates.
    amp_threshold : float, optional
        Standard deviation multiplier for amplifier trigger.
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        vocab_size=None,
        disallowed_tokens=None,
        replay_size=500,
        bias_decay: float = 0.999,
        bias_max: float = 5.0,
        gate_use_classifier: bool = False,
        gate_classifier_hidden: int = 32,
        gate_classifier_scale: float = 5.0,
        amp_rate: float = 0.1,
        amp_threshold: float = 2.0,
    ):
        super().__init__(
            hidden_size,
            output_size,
            vocab_size=vocab_size,
            disallowed_tokens=disallowed_tokens,
            replay_size=replay_size,
            bias_decay=bias_decay,
            bias_max=bias_max,
            gate_use_classifier=gate_use_classifier,
            gate_classifier_hidden=gate_classifier_hidden,
            gate_classifier_scale=gate_classifier_scale,
            amp_rate=amp_rate,
            amp_threshold=amp_threshold,
        )

    def forward(self, hidden):
        return self.core_forward(hidden)

    def training_step(self, hidden, target, optimizer, criterion, *, lambda_reg: float = 0.0):
        self.train()
        optimizer.zero_grad()
        with torch.no_grad():
            self.anchor_bias.mul_(self.bias_decay)
            torch.clamp_(self.anchor_bias, -self.bias_max, self.bias_max)
        logits, raw_logits, final_hidden = self.forward(hidden)
        return self._shared_training_logic(final_hidden, logits, target, optimizer, criterion, lambda_reg=lambda_reg)


from .integration import attach_genesis_plugin

__all__ = [
    "SelfReplayBuffer",
    "EthicalGate",
    "GenesisCore",
    "GenesisPlugin",
    "IntegratedLearningModule",
    "attach_genesis_plugin",
]
