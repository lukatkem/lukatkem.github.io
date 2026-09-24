"""gradia tests — gradient checks vs finite differences, classic identities,
training end-to-end, determinism.

The workhorse is ``assert_matches_numeric``: it builds an expression twice,
once through the engine (backward) and once through central finite differences,
and requires the two to agree. Every op is checked at a couple of points; the
losses are checked against their hand-derived formulas; and the two training
tests prove the whole stack learns real tasks.
"""
from __future__ import annotations

import math
from typing import Callable, List, Tuple

import pytest

from gradia import (
    MLP,
    LCG,
    SGD,
    Tensor,
    accuracy,
    cross_entropy_loss,
    mse_loss,
    softmax_cross_entropy,
    train,
)

EPS = 1e-6
TOL = 1e-5


def close(a: float, b: float, tol: float = 1e-12) -> None:
    assert abs(a - b) < tol, f"{a!r} vs {b!r} (diff {abs(a - b):.3e} >= {tol:.0e})"


def autograd_grad(f: Callable[[Tensor], Tensor], x: float) -> float:
    t = Tensor(x)
    f(t).backward()
    return t.grad


def numeric_grad(f: Callable[[Tensor], Tensor], x: float, eps: float = EPS) -> float:
    return (f(Tensor(x + eps)).data - f(Tensor(x - eps)).data) / (2.0 * eps)


def assert_matches_numeric(f: Callable[[Tensor], Tensor], x: float, tol: float = TOL) -> None:
    got, want = autograd_grad(f, x), numeric_grad(f, x)
    assert abs(got - want) < tol, f"at x={x}: autograd {got!r} vs numeric {want!r}"


def fit_x2(epochs: int, lr: float) -> Tuple[MLP, List[Tuple[int, float]]]:
    """The x² task from the demo, as a reusable fixture for the tests."""
    pts = [(-2.0 + 4.0 * i / 15.0) for i in range(16)]
    xs = [[Tensor(x), Tensor(1.0)] for x in pts]
    ys = [x * x for x in pts]
    model = MLP([2, 8, 1], activation="tanh")
    history = train(model, xs, ys, epochs=epochs, lr=lr, loss_fn=mse_loss)
    return model, history


def blob_points(n_per_class: int = 20, std: float = 0.6, seed: int = 7):
    """The deterministic gaussian blobs from the demo."""
    rng = LCG(seed)
    points, labels = [], []
    for label, center in enumerate(((-1.0, -1.0), (1.0, 1.0))):
        for _ in range(n_per_class):
            points.append([Tensor(rng.normal(center[0], std)), Tensor(rng.normal(center[1], std))])
            labels.append(label)
    return points, labels


def train_blobs(epochs: int = 300, lr: float = 0.1) -> Tuple[List[List[Tensor]], List[int], float]:
    points, labels = blob_points()
    model = MLP([2, 8, 8, 2], activation="tanh")
    train(model, points, labels, epochs=epochs, lr=lr, loss_fn=cross_entropy_loss)
    return [model(p) for p in points], labels, model


# ---------- per-op gradient checks ----------

def test_add_gradient():
    assert_matches_numeric(lambda x: x + 3.0, 0.5)
    assert_matches_numeric(lambda x: x + 3.0, -1.25)
    # x + x must accumulate BOTH branches: d/dx = 2
    t = Tensor(0.7)
    (t + t).backward()
    close(t.grad, 2.0)


def test_mul_gradient():
    assert_matches_numeric(lambda x: x * 3.0, 0.5)
    assert_matches_numeric(lambda x: x * x, 1.5)
    assert_matches_numeric(lambda x: x * x, -2.0)


def test_sub_and_neg_gradient():
    assert_matches_numeric(lambda x: x - 3.0, 0.5)
    assert_matches_numeric(lambda x: 3.0 - x, -0.75)  # reflected sub
    assert_matches_numeric(lambda x: -(x * x), 0.8)
    t = Tensor(2.0)
    (-t).backward()
    close(t.grad, -1.0)


def test_div_gradient_matches_pow_identity():
    # a / b is a * b**-1 by construction; the gradient must follow.
    assert_matches_numeric(lambda x: x / 2.0, 1.5)
    assert_matches_numeric(lambda x: 1.0 / x, 2.0)   # reflected div: -1/x²
    assert_matches_numeric(lambda x: 1.0 / x, 4.0)
    t = Tensor(3.0)
    (t / 4.0).backward()
    close(t.grad, 1.0 / 4.0, tol=1e-12)


def test_pow_gradient():
    assert_matches_numeric(lambda x: x ** 3, 1.5)
    assert_matches_numeric(lambda x: x ** 3, -1.25)  # odd power, negative base
    assert_matches_numeric(lambda x: x ** 0.5, 2.0)  # sqrt
    assert_matches_numeric(lambda x: x ** -1, 3.0)


def test_tanh_gradient():
    assert_matches_numeric(lambda x: x.tanh(), 0.5)
    assert_matches_numeric(lambda x: x.tanh(), -1.3)


def test_exp_gradient():
    assert_matches_numeric(lambda x: x.exp(), 0.3)
    assert_matches_numeric(lambda x: x.exp(), -0.7)


def test_log_gradient():
    assert_matches_numeric(lambda x: x.log(), 1.5)
    assert_matches_numeric(lambda x: x.log(), 4.0)
    with pytest.raises(ValueError):
        Tensor(0.0).log()
    with pytest.raises(ValueError):
        Tensor(-1.5).log()


def test_relu_gradient_and_values():
    assert_matches_numeric(lambda x: x.relu(), 0.9)
    assert_matches_numeric(lambda x: x.relu(), -0.7)
    close(Tensor(3.0).relu().data, 3.0)
    close(Tensor(-2.0).relu().data, 0.0)
    # at zero the subgradient is defined here as 0 (out.data > 0 gate)
    t = Tensor(0.0)
    t.relu().backward()
    close(t.grad, 0.0)


def test_sigmoid_gradient_and_stability():
    assert_matches_numeric(lambda x: x.sigmoid(), 0.5)
    assert_matches_numeric(lambda x: x.sigmoid(), -2.5)
    # the sign-split implementation must not overflow at either extreme
    close(Tensor(-800.0).sigmoid().data, 0.0, tol=1e-12)
    close(Tensor(800.0).sigmoid().data, 1.0, tol=1e-12)


def test_composite_chain_rule():
    # y = tanh(a*b + c) — the manual calculus: dy/da = b·(1-y²), etc.
    a, b, c = Tensor(0.7), Tensor(-1.3), Tensor(0.4)
    y = (a * b + c).tanh()
    y.backward()
    one_minus = 1.0 - math.tanh(0.7 * -1.3 + 0.4) ** 2
    close(a.grad, -1.3 * one_minus)
    close(b.grad, 0.7 * one_minus)
    close(c.grad, one_minus)
    # and the same expression checked numerically, arg by arg
    assert_matches_numeric(lambda x: (x * Tensor(-1.3) + 0.4).tanh(), 0.7)
    assert_matches_numeric(lambda x: (Tensor(0.7) * x + 0.4).tanh(), -1.3)
    assert_matches_numeric(lambda x: (Tensor(0.7) * Tensor(-1.3) + x).tanh(), 0.4)


def test_backward_seeds_root_and_leaves_strangers_alone():
    x = Tensor(2.0)
    y = Tensor(5.0)          # not part of the graph below
    z = (x * 3.0).tanh()
    z.backward()
    close(z.grad, 1.0)
    close(x.grad, 3.0 * (1.0 - math.tanh(6.0) ** 2))
    close(y.grad, 0.0)       # untouched — it never fed z


def test_sum_mean_max_gradients():
    a, b, c = Tensor(1.0), Tensor(2.0), Tensor(3.0)
    Tensor.sum([a, b, c]).backward()
    close(a.grad, 1.0)
    close(b.grad, 1.0)
    close(c.grad, 1.0)
    a, b, c = Tensor(1.0), Tensor(2.0), Tensor(3.0)
    Tensor.mean([a, b, c]).backward()
    close(a.grad, 1.0 / 3.0)
    close(b.grad, 1.0 / 3.0)
    close(c.grad, 1.0 / 3.0)
    a, b, c = Tensor(1.0), Tensor(3.5), Tensor(2.0)
    m = Tensor.max([a, b, c])
    close(m.data, 3.5)
    m.backward()
    close(b.grad, 1.0)       # the winner only
    close(a.grad, 0.0)
    close(c.grad, 0.0)
    # ties go to the lowest index, deterministically
    t1, t2 = Tensor(2.0), Tensor(2.0)
    Tensor.max([t1, t2]).backward()
    close(t1.grad, 1.0)
    close(t2.grad, 0.0)


def test_pow_and_log_input_validation():
    with pytest.raises(TypeError):
        Tensor(2.0) ** Tensor(3.0)          # pow is scalar-only
    with pytest.raises(TypeError):
        Tensor(2.0) ** "3"                  # type: ignore[operator]
    with pytest.raises(ValueError):
        Tensor(-2.0) ** 0.5                 # not real
    t = Tensor(-2.0)
    (t ** 3).backward()                     # integer power of a negative base is fine
    close(t.grad, 12.0)                     # d/dx x³ = 3x² = 12


def test_repr_shows_data_and_grad():
    t = Tensor(0.5)
    r = repr(t)
    assert "0.5" in r and "Tensor" in r
    (t * 2.0).backward()
    assert "grad=2" in repr(t)              # the accumulated grad is visible


# ---------- losses ----------

def test_mse_loss_gradient_formula():
    preds = [Tensor(1.0), Tensor(2.0), Tensor(3.0)]
    targets = [1.5, 1.0, 2.0]               # plain floats are legal targets
    loss = mse_loss(preds, targets)
    close(loss.data, ((0.5 ** 2) + (1.0 ** 2) + (1.0 ** 2)) / 3.0)
    loss.backward()
    for p, t in zip(preds, targets):
        close(p.grad, 2.0 * (p.data - t) / 3.0)   # d/dp mean((p-t)²) = 2(p-t)/n
    with pytest.raises(ValueError):
        mse_loss([Tensor(1.0)], [])


def test_softmax_cross_entropy_identity():
    # The classic identity: dCE/dlogit_i = softmax_i - onehot_i
    logits = [Tensor(0.3), Tensor(-1.2), Tensor(2.4)]
    y = 1
    ce = softmax_cross_entropy(logits, y)
    exps = [math.exp(l.data - max(l.data for l in logits)) for l in logits]
    z = sum(exps)
    softmax = [e / z for e in exps]
    close(ce.data, -math.log(softmax[y]))   # the value is -log softmax_y
    ce.backward()
    for i, l in enumerate(logits):
        want = softmax[i] - (1.0 if i == y else 0.0)
        close(l.grad, want)
    # shift invariance: adding a constant to every logit changes nothing
    shifted = [Tensor(l.data + 100.0) for l in logits]
    close(softmax_cross_entropy(shifted, y).data, ce.data)


def test_cross_entropy_loss_is_batch_mean():
    a = [Tensor(0.3), Tensor(-1.2)]
    b = [Tensor(0.1), Tensor(0.4)]
    batch = cross_entropy_loss([a, b], [0, 1])
    want = (softmax_cross_entropy(a, 0).data + softmax_cross_entropy(b, 1).data) / 2.0
    close(batch.data, want)
    batch.backward()
    close(a[0].grad, 0.5 * (_softmax(a)[0] - 1.0))   # mean divides the identity by n


def _softmax(logits: List[Tensor]) -> List[float]:
    exps = [math.exp(l.data - max(l.data for l in logits)) for l in logits]
    z = sum(exps)
    return [e / z for e in exps]


# ---------- network pieces ----------

def test_mlp_parameter_count_and_forward_shape():
    model = MLP([2, 8, 1])
    # (2·8 + 8) + (8·1 + 1) = 24 + 9
    assert len(model.parameters()) == 33
    out = model([Tensor(0.5), Tensor(-0.5)])
    assert len(out) == 1 and isinstance(out[0], Tensor)
    out8 = MLP([2, 8, 8, 2])([Tensor(0.5), Tensor(-0.5)])
    assert len(out8) == 2


def test_initialization_deterministic_and_scaled():
    a, b = MLP([2, 8, 1]), MLP([2, 8, 1])           # same default seed
    assert [p.data for p in a.parameters()] == [p.data for p in b.parameters()]
    c = MLP([2, 8, 1], seed=99)
    assert [p.data for p in a.parameters()] != [p.data for p in c.parameters()]
    for layer, n_in in zip(a.layers, (2, 8)):
        for neuron in layer.neurons:
            for w in neuron.weights:
                assert abs(w.data) <= 1.0 / math.sqrt(n_in) + 1e-12
    # biases start at zero
    for layer in a.layers:
        for neuron in layer.neurons:
            assert neuron.bias.data == 0.0


def test_lcg_is_deterministic_and_in_range():
    a, b = LCG(42), LCG(42)
    seq_a = [a.next_float() for _ in range(100)]
    seq_b = [b.next_float() for _ in range(100)]
    assert seq_a == seq_b
    assert all(0.0 < v <= 1.0 for v in seq_a)       # never zero → log-safe
    assert seq_a != [LCG(43).next_float() for _ in range(100)]
    points, _ = blob_points()                        # Box–Muller path runs clean
    assert len(points) == 40


def test_unknown_activation_rejected():
    # sizes [2, 4, 2]: the middle width guarantees a hidden layer exists, so
    # the bad name is actually resolved (a bare [2, 2] has no hidden layers).
    with pytest.raises(ValueError):
        MLP([2, 4, 2], activation="gelu")


def test_zero_grad_clears_everything():
    model = MLP([2, 4, 1])
    loss = mse_loss([model([Tensor(1.0), Tensor(2.0)])[0]], [1.0])
    loss.backward()
    assert any(p.grad != 0.0 for p in model.parameters())
    model.zero_grad()
    assert all(p.grad == 0.0 for p in model.parameters())
    # the SGD flavor agrees
    loss = mse_loss([model([Tensor(1.0), Tensor(2.0)])[0]], [1.0])
    loss.backward()
    opt = SGD(model.parameters(), 0.1)
    opt.zero_grad()
    assert all(p.grad == 0.0 for p in model.parameters())


def test_sgd_step_descends_a_bowl():
    w = Tensor(0.0)
    opt = SGD([w], lr=0.1)
    for _ in range(10):
        loss = (w - 2.0) ** 2.0
        loss.backward()
        opt.step()
    assert abs(w.data - 2.0) < 0.25                 # 2 - 2·0.8¹⁰ ≈ 0.21 away
    # step(lr) overrides the stored rate
    w2 = Tensor(0.0)
    opt2 = SGD([w2], lr=0.0)
    ((w2 - 2.0) ** 2.0).backward()
    opt2.step(0.5)
    assert w2.data == pytest.approx(2.0)            # grad -4 → w2 = 0 - 0.5·(-4)


# ---------- end-to-end training ----------

def test_train_history_shape():
    model = MLP([2, 4, 1])
    xs = [[Tensor(0.5), Tensor(1.0)], [Tensor(-1.0), Tensor(0.5)]]
    ys = [1.0, 0.25]
    history = train(model, xs, ys, epochs=7, lr=0.05)
    assert len(history) == 7
    assert [e for e, _ in history] == list(range(1, 8))
    assert all(isinstance(v, float) for _, v in history)
    with pytest.raises(ValueError):
        train(model, xs, ys[:-1], epochs=5, lr=0.05)


def test_mlp_trains_x2_below_threshold():
    model, history = fit_x2(epochs=600, lr=0.05)
    assert history[-1][1] < 0.05                    # the demo's promise, tested
    assert history[-1][1] < history[0][1]           # and it actually went down
    for x in (-2.0, -1.0, 0.0, 1.0, 2.0):
        pred = model([Tensor(x), Tensor(1.0)])[0].data
        assert abs(pred - x * x) < 0.35


def test_classifier_reaches_high_accuracy():
    logits, labels, _ = train_blobs()
    acc = accuracy(logits, labels)
    assert acc >= 0.9


def test_two_identical_trainings_are_bit_identical():
    _, h1 = fit_x2(epochs=80, lr=0.05)
    _, h2 = fit_x2(epochs=80, lr=0.05)
    assert h1 == h2
    # the classifier path agrees too (CE loss, deeper net)
    pts, labels = blob_points()
    m1, m2 = MLP([2, 8, 8, 2]), MLP([2, 8, 8, 2])
    t1 = train(m1, pts, labels, epochs=25, lr=0.1, loss_fn=cross_entropy_loss)
    t2 = train(m2, pts, labels, epochs=25, lr=0.1, loss_fn=cross_entropy_loss)
    assert t1 == t2
    assert [p.data for p in m1.parameters()] == [p.data for p in m2.parameters()]
