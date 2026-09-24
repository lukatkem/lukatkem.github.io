"""A tiny neural-network library on top of the engine — plus losses and SGD.

Modules are plain containers of Tensors: ``parameters()`` lists every
learnable Tensor, ``zero_grad()`` clears the gradients. Weights come from a
fixed-seed LCG (the ``random`` module is off-limits) — uniform in [-1, 1]
scaled by 1/sqrt(n_in), the classic heuristic that keeps pre-activations near
the linear zone of tanh at initialization. Everything — init, forward, loss,
update — is pure python floats, so two runs with the same seed agree bit for
bit.

The losses are built from engine ops, not formulas handed to the optimizer:
``mse_loss`` is a mean of squared diffs, and ``softmax_cross_entropy`` is
``log(sum(exp(logits))) - correct_logit``. Backpropagating through that
composition reproduces the textbook gradients on its own — including the
classic ``softmax - onehot`` identity the tests check directly.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

from .engine import Number, Tensor

_TWO_PI = 2.0 * math.pi


class LCG:
    """A linear congruential generator with glibc's classic constants.

    State lives in [0, 2**31); ``next_float`` returns values in (0, 1] — never
    zero, so ``log`` is always safe. Deterministic by construction: same seed,
    same sequence, on every machine and every run.
    """

    A = 1103515245
    C = 12345
    M = 2 ** 31

    def __init__(self, seed: int) -> None:
        self.state = seed % self.M

    def next_float(self) -> float:
        self.state = (self.state * self.A + self.C) % self.M
        return (self.state + 1.0) / (self.M + 1.0)

    def uniform(self, lo: float = 0.0, hi: float = 1.0) -> float:
        return lo + (hi - lo) * self.next_float()

    def normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        """One Box–Muller sample (the cos half of each uniform pair)."""
        u1, u2 = self.next_float(), self.next_float()
        return mean + std * math.sqrt(-2.0 * math.log(u1)) * math.cos(_TWO_PI * u2)


def _resolve_activation(name: str) -> Optional[Callable[[Tensor], Tensor]]:
    """Map an activation name to its engine method; "linear" means identity."""
    activations: Dict[str, Optional[Callable[[Tensor], Tensor]]] = {
        "tanh": Tensor.tanh,
        "relu": Tensor.relu,
        "sigmoid": Tensor.sigmoid,
        "linear": None,
    }
    if name not in activations:
        raise ValueError(
            f"unknown activation {name!r} — expected one of: {', '.join(sorted(activations))}"
        )
    return activations[name]


class Module:
    """Base class for anything that owns learnable Tensors."""

    def parameters(self) -> List[Tensor]:
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    """One unit: act(w·x + b), with weights drawn scaled by 1/sqrt(n_in)."""

    def __init__(self, n_in: int, activation: str = "tanh", rng: Optional[LCG] = None) -> None:
        if isinstance(n_in, bool) or not isinstance(n_in, int) or n_in < 1:
            raise ValueError(f"n_in must be a positive integer, got {n_in!r}")
        self.n_in = n_in
        self.activation = activation
        self._act = _resolve_activation(activation)
        rng = rng if rng is not None else LCG(0)
        scale = 1.0 / math.sqrt(n_in)
        self.weights = [Tensor(rng.uniform(-1.0, 1.0) * scale) for _ in range(n_in)]
        self.bias = Tensor(0.0)

    def __call__(self, xs: Sequence[Union[Tensor, float]]) -> Tensor:
        xs = list(xs)
        if len(xs) != self.n_in:
            raise ValueError(f"neuron expects {self.n_in} inputs, got {len(xs)}")
        z = self.bias + Tensor.sum([w * x for w, x in zip(self.weights, xs)])
        return z if self._act is None else self._act(z)

    def parameters(self) -> List[Tensor]:
        return self.weights + [self.bias]


class Layer(Module):
    """n_out neurons sharing one input vector, drawing from one LCG in order."""

    def __init__(self, n_in: int, n_out: int, activation: str = "tanh",
                 rng: Optional[LCG] = None) -> None:
        if isinstance(n_out, bool) or not isinstance(n_out, int) or n_out < 1:
            raise ValueError(f"n_out must be a positive integer, got {n_out!r}")
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        rng = rng if rng is not None else LCG(0)
        self.neurons = [Neuron(n_in, activation, rng) for _ in range(n_out)]

    def __call__(self, xs: Sequence[Union[Tensor, float]]) -> List[Tensor]:
        return [neuron(xs) for neuron in self.neurons]

    def parameters(self) -> List[Tensor]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    """A stack of fully-connected layers.

    Hidden layers use ``activation``; the output layer is linear by default
    (``output_activation=None``) — regression and softmax-classification both
    want raw logits, so the caller opts into a squashing output explicitly.
    One LCG seeded with ``seed`` feeds every layer in order, which makes the
    whole initialization reproducible from a single number.
    """

    def __init__(self, sizes: Sequence[int], activation: str = "tanh",
                 output_activation: Optional[str] = None, seed: int = 1337) -> None:
        sizes = list(sizes)
        if len(sizes) < 2 or any(
            isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in sizes
        ):
            raise ValueError(f"sizes must be integers >= 1 with at least two entries, got {sizes!r}")
        self.sizes = sizes
        rng = LCG(seed)
        self.layers = [Layer(a, b, activation, rng) for a, b in zip(sizes, sizes[1:-1])]
        self.layers.append(Layer(sizes[-2], sizes[-1], output_activation or "linear", rng))

    def __call__(self, xs: Sequence[Union[Tensor, float]]) -> List[Tensor]:
        for layer in self.layers:
            xs = layer(xs)
        return list(xs)

    def parameters(self) -> List[Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]


class SGD:
    """Stochastic gradient descent, batch-flavored: p -= lr * p.grad."""

    def __init__(self, params: Iterable[Tensor], lr: float = 0.05) -> None:
        self.params = list(params)
        self.lr = float(lr)

    def step(self, lr: Optional[float] = None) -> None:
        rate = self.lr if lr is None else float(lr)
        for p in self.params:
            p.data -= rate * p.grad

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0


# ---------- losses ----------
#
# Both take a batch: preds is a list of output Tensors (one per sample), and
# targets are either Tensors or plain floats. Plain floats are fine — the
# engine wraps constants on the fly, and only preds need gradients anyway.

def mse_loss(preds: Sequence[Union[Tensor, float]],
             targets: Sequence[Union[Tensor, float]]) -> Tensor:
    """Mean squared error over the batch: mean((pred - target)^2)."""
    p, t = list(preds), list(targets)
    if not p or len(p) != len(t):
        raise ValueError(f"mse_loss needs equal-length non-empty lists, got {len(p)} vs {len(t)}")
    return Tensor.mean([(pi - ti) ** 2.0 for pi, ti in zip(p, t)])


def softmax_cross_entropy(logits: Sequence[Tensor], class_index: int) -> Tensor:
    """Cross-entropy of one sample: log(sum(exp(logits))) - logits[class_index].

    Built from the engine's sub/exp/sum/log ops, so backprop through it
    yields exactly ``softmax(logits) - onehot(class_index)`` — the identity
    the tests verify against a hand-computed softmax. The value uses the
    log-sum-exp identity ``lse(x) = peak + log(sum(exp(x - peak)))``; the
    peak shift is a plain float constant (softmax is shift-invariant), which
    keeps exp() from overflowing on confident logits without touching any
    gradient.
    """
    ls = list(logits)
    if not ls:
        raise ValueError("softmax_cross_entropy needs at least one logit")
    if isinstance(class_index, bool) or not isinstance(class_index, int) \
            or not 0 <= class_index < len(ls):
        raise ValueError(f"class_index must be an integer in [0, {len(ls) - 1}], got {class_index!r}")
    peak = max(l.data for l in ls)
    exps = [(l - peak).exp() for l in ls]
    return Tensor.sum(exps).log() + peak - ls[class_index]


def cross_entropy_loss(logits_batch: Sequence[Sequence[Tensor]],
                       class_indices: Sequence[int]) -> Tensor:
    """Batch wrapper: mean softmax_cross_entropy over per-sample logits."""
    lb, ci = list(logits_batch), list(class_indices)
    if not lb or len(lb) != len(ci):
        raise ValueError(
            f"cross_entropy_loss needs equal-length non-empty lists, got {len(lb)} vs {len(ci)}"
        )
    return Tensor.mean([softmax_cross_entropy(lg, y) for lg, y in zip(lb, ci)])


def accuracy(logits_batch: Sequence[Sequence[Tensor]], labels: Sequence[int]) -> float:
    """Fraction of samples whose highest logit lands on the true class.

    Ties go to the lowest index, same as ``Tensor.max``.
    """
    lb, ls = list(logits_batch), list(labels)
    if not lb or len(lb) != len(ls):
        raise ValueError(
            f"accuracy needs equal-length non-empty lists, got {len(lb)} vs {len(ls)}"
        )
    correct = 0
    for logits, label in zip(lb, ls):
        best = 0
        for i in range(1, len(logits)):
            if logits[i].data > logits[best].data:
                best = i
        if best == label:
            correct += 1
    return correct / len(lb)
