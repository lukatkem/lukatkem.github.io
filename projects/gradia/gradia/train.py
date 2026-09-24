"""The training loop — full-batch gradient descent, nothing up its sleeve.

Each epoch rebuilds the graph by running the forward pass, computes one loss
for the whole batch, calls ``loss.backward()`` (the engine walks that fresh
graph and deposits gradients into every parameter), and takes one SGD step.
The graph is created by the forward pass and dropped when the next epoch
overwrites it — in reverse-mode autodiff, the forward pass *is* the graph.

The loop is silent: it returns the loss history and leaves printing to the
caller. Determinism follows from the engine's — every epoch sees the same
samples in the same order, updates in a fixed order, and no randomness
anywhere.
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Union

from .engine import Tensor
from .nn import SGD, mse_loss

LossFn = Callable[[Sequence[Tensor], Sequence[Union[Tensor, float]]], Tensor]


def train(model, xs, ys, epochs: int, lr: float, loss_fn: LossFn = mse_loss) -> List[Tuple[int, float]]:
    """Fit ``model`` to (xs, ys) with full-batch gradient descent.

    The model is called once per sample and must return a list of output
    Tensors (single-output models are unwrapped automatically — see the loop).
    ``loss_fn(preds, ys)`` folds the batch into one scalar Tensor, typically
    ``mse_loss`` or ``cross_entropy_loss``.

    Returns the history as a list of ``(epoch, loss_value)`` tuples, one per
    epoch, with epochs counting from 1. Same model, same data, same settings
    — same history, bit for bit.
    """
    xs, ys = list(xs), list(ys)
    if not xs or len(xs) != len(ys):
        raise ValueError(f"train needs equal-length non-empty xs and ys, got {len(xs)} vs {len(ys)}")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    lr = float(lr)
    opt = SGD(model.parameters(), lr)
    history: List[Tuple[int, float]] = []
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        preds = []
        for x in xs:
            outs = model(x)
            # Convention: models return one output Tensor per sample position.
            # Single-output models are unwrapped so mse-style losses see a
            # flat list of Tensors; multi-output models keep their logit
            # vectors for cross_entropy_loss.
            if isinstance(outs, Tensor):
                preds.append(outs)
            elif len(outs) == 1:
                preds.append(outs[0])
            else:
                preds.append(list(outs))
        loss = loss_fn(preds, ys)
        loss.backward()
        opt.step(lr)
        history.append((epoch, loss.data))
    return history
