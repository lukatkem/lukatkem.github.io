"""A self-contained walkthrough: one gradient check, one regression, one classifier.

Three demonstrations, no arguments, same numbers every run:

1. backward sanity — autograd's dy/dx against a central finite difference;
2. a 2→8→1 MLP fitting y = x² over [-2, 2] (the linear output layer is the
   default; the constant second input just gives the 2-wide layer a bias
   channel, since the task itself is one-dimensional);
3. a 2→8→8→2 classifier separating two deterministic gaussian blobs.
"""
from __future__ import annotations

from typing import List, Tuple

from .engine import Tensor
from .nn import MLP, LCG, accuracy, cross_entropy_loss, mse_loss
from .train import train

_REG_EPOCHS = 600
_REG_LR = 0.05
_CLF_EPOCHS = 300
_CLF_LR = 0.10


def _print_history(history: List[Tuple[int, float]], label: str) -> None:
    for epoch, loss in history:
        if epoch == 1 or epoch % 100 == 0:
            print(f"epoch {epoch:>4} · {label} {loss:.6f}")


def _demo_backward() -> None:
    print("== 1 · backward sanity — the chain rule vs finite differences ==\n")
    x0 = 0.5

    def f(x: Tensor) -> Tensor:
        return (x * 2.0 + 1.0).tanh()

    x = Tensor(x0)
    y = f(x)
    y.backward()
    eps = 1e-6
    numeric = (f(Tensor(x0 + eps)).data - f(Tensor(x0 - eps)).data) / (2.0 * eps)
    print("y = (x·2 + 1).tanh()  at x = 0.5")
    print(f"  y                    = {y.data:.9f}")
    print(f"  dy/dx (autograd)     = {x.grad:.9f}")
    print(f"  dy/dx (finite diff)  = {numeric:.9f}")
    print(f"  difference           = {abs(x.grad - numeric):.2e}")


def _demo_regression() -> None:
    print("\n== 2 · regression — a 2→8→1 MLP fits y = x² over [-2, 2] ==\n")
    points = [(-2.0 + 4.0 * i / 15.0) for i in range(16)]
    points = [(x, x * x) for x in points]
    xs = [[Tensor(x), Tensor(1.0)] for x, _ in points]
    ys = [y for _, y in points]
    model = MLP([2, 8, 1], activation="tanh")
    history = train(model, xs, ys, epochs=_REG_EPOCHS, lr=_REG_LR, loss_fn=mse_loss)
    _print_history(history, "mse")
    print()
    for x in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
        pred = model([Tensor(x), Tensor(1.0)])[0].data
        print(f"  x = {x:+.1f} · predicted {pred:+.4f} · true {x * x:+.4f}")


def _demo_classifier() -> None:
    print("\n== 3 · classification — two gaussian blobs, 2→8→8→2 with softmax CE ==\n")
    rng = LCG(7)
    points: List[List[Tensor]] = []
    labels: List[int] = []
    for label, center in enumerate(((-1.0, -1.0), (1.0, 1.0))):
        for _ in range(20):
            points.append([Tensor(rng.normal(center[0], 0.6)),
                           Tensor(rng.normal(center[1], 0.6))])
            labels.append(label)
    model = MLP([2, 8, 8, 2], activation="tanh")
    history = train(model, points, labels, epochs=_CLF_EPOCHS, lr=_CLF_LR,
                    loss_fn=cross_entropy_loss)
    _print_history(history, "cross-entropy")
    logits = [model(p) for p in points]
    acc = accuracy(logits, labels)
    print(f"\n  final accuracy: {acc:.0%} on {len(points)} blob points")


def main() -> int:
    print("== gradia demo — reverse-mode autodiff, end to end ==\n")
    _demo_backward()
    _demo_regression()
    _demo_classifier()
    print("\ndone — pure stdlib, fully deterministic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
