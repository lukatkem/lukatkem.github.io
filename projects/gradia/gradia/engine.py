"""The autograd core — scalar reverse-mode automatic differentiation.

Every operation on a Tensor records what produced it: the parent Tensors in
``_prev``, an operator tag in ``_op``, and a ``_backward`` closure that knows
how to push the output's gradient back onto its parents. Calling ``backward()``
on a result seeds its gradient with 1.0, walks the recorded graph in reverse
topological order, and applies the chain rule one local derivative at a time —
the same three steps ``loss.backward()`` performs in PyTorch, with nothing
hidden.

Tensors are scalars (a single float) — no shapes, no broadcasting. Two
consequences are deliberate: every gradient rule is visible in a two-line
closure, and traversal is fully deterministic. ``_prev`` is a set, and set
iteration order follows memory addresses, so ``backward()`` instead orders the
graph by creation serial: every op builds its output *after* its inputs, so
ascending serial number is always a valid topological order, independent of
where CPython happened to place the objects. Same graph, same gradients, bit
for bit, every run.
"""
from __future__ import annotations

import itertools
import math
import operator
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Set, Union

Number = Union[int, float]

_SERIAL: Iterator[int] = itertools.count()

_serial_key = operator.attrgetter("_serial")


def _noop() -> None:
    """Default ``_backward`` for leaf tensors: nothing to propagate."""


def _coerce(value: Union["Tensor", Number]) -> "Tensor":
    """Wrap plain numbers as constant Tensors; pass Tensors through unchanged."""
    if isinstance(value, Tensor):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a Tensor or real number, got {type(value).__name__}")
    return Tensor(float(value))


@dataclass(eq=False)
class Tensor:
    """A scalar value that remembers how it was made.

    ``data`` is the forward value, ``grad`` the accumulated d(output)/d(this).
    ``_prev`` holds the parent Tensors, ``_op`` tags the operation that built
    this node, and ``_backward`` is the closure that distributes ``grad`` to
    the parents. ``eq=False`` keeps identity equality and hashing, which is
    what the graph's sets (and ``backward``'s visited check) rely on.
    """

    data: float
    grad: float = 0.0
    _backward: Callable[[], None] = field(default=_noop, repr=False, compare=False)
    _prev: Set["Tensor"] = field(default_factory=set, repr=False, compare=False)
    _op: str = field(default="", repr=False, compare=False)
    _serial: int = field(default_factory=lambda: next(_SERIAL), repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.data, bool) or not isinstance(self.data, (int, float)):
            raise TypeError(f"Tensor data must be a real number, got {type(self.data).__name__}")
        self.data = float(self.data)

    # ---------- graph traversal ----------

    def backward(self) -> None:
        """Backpropagate through the whole graph, starting from this node.

        Seeds ``self.grad = 1.0`` (overwriting whatever was there), collects
        every ancestor, and calls each node's ``_backward`` in reverse
        topological order — so by the time a node speaks, every consumer has
        already deposited its contribution into ``node.grad``. Nodes are
        processed newest-first by creation serial, which is a topological
        order by construction (see the module docstring).
        """
        stack: List[Tensor] = [self]
        seen = {id(self)}
        nodes: List[Tensor] = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            for parent in node._prev:
                if id(parent) not in seen:
                    seen.add(id(parent))
                    stack.append(parent)
        nodes.sort(key=_serial_key)
        self.grad = 1.0
        for node in reversed(nodes):
            node._backward()

    def zero_grad(self) -> None:
        """Reset this tensor's accumulated gradient to zero."""
        self.grad = 0.0

    # ---------- arithmetic ----------

    def __add__(self, other: Union["Tensor", Number]) -> "Tensor":
        other = _coerce(other)
        out = Tensor(self.data + other.data, _op="+", _prev={self, other})

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: Union["Tensor", Number]) -> "Tensor":
        other = _coerce(other)
        out = Tensor(self.data - other.data, _op="-", _prev={self, other})

        def _backward() -> None:
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Union["Tensor", Number]) -> "Tensor":
        other = _coerce(other)
        out = Tensor(self.data * other.data, _op="*", _prev={self, other})

        def _backward() -> None:
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __truediv__(self, other: Union["Tensor", Number]) -> "Tensor":
        # a / b is a * b**-1 — no special-case gradient needed; the pow and
        # mul rules compose and the graph stays honest about it.
        return self * _coerce(other) ** -1.0

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, _op="neg", _prev={self})

        def _backward() -> None:
            self.grad -= out.grad

        out._backward = _backward
        return out

    def __pow__(self, power: Number) -> "Tensor":
        if isinstance(power, Tensor):
            raise TypeError("pow is scalar-only — exponent must be an int or float")
        if isinstance(power, bool) or not isinstance(power, (int, float)):
            raise TypeError(f"pow is scalar-only, got exponent of type {type(power).__name__}")
        p = float(power)
        if self.data < 0.0 and p != int(p):
            raise ValueError(f"negative base {self.data} with non-integer power {power} is not real")
        out = Tensor(self.data ** p, _op="pow", _prev={self})

        def _backward() -> None:
            self.grad += out.grad * p * self.data ** (p - 1.0)

        out._backward = _backward
        return out

    # reflected ops — plain numbers on the left
    def __radd__(self, other: Number) -> "Tensor":
        return _coerce(other) + self

    def __rsub__(self, other: Number) -> "Tensor":
        return _coerce(other) - self

    def __rmul__(self, other: Number) -> "Tensor":
        return _coerce(other) * self

    def __rtruediv__(self, other: Number) -> "Tensor":
        return _coerce(other) * self ** -1.0

    # ---------- functions ----------

    def tanh(self) -> "Tensor":
        t = math.tanh(self.data)
        out = Tensor(t, _op="tanh", _prev={self})

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        e = math.exp(self.data)
        out = Tensor(e, _op="exp", _prev={self})

        def _backward() -> None:
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        if self.data <= 0.0:
            raise ValueError(f"log needs a positive value, got {self.data}")
        out = Tensor(math.log(self.data), _op="log", _prev={self})

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(self.data if self.data > 0.0 else 0.0, _op="relu", _prev={self})

        def _backward() -> None:
            if out.data > 0.0:
                self.grad += out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        # Split by sign so exp() only ever sees non-positive arguments —
        # no overflow at either extreme.
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x)) if x >= 0.0 else math.exp(x) / (1.0 + math.exp(x))
        out = Tensor(s, _op="sigmoid", _prev={self})

        def _backward() -> None:
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    # ---------- reductions over lists of Tensors ----------
    #
    # Tensors are scalars, so a reduction takes an iterable and fuses it into
    # ONE graph node — cheaper and cleaner than chaining binary adds.

    @staticmethod
    def sum(tensors: Iterable["Tensor"]) -> "Tensor":
        ts = list(tensors)
        if not ts:
            raise ValueError("sum needs at least one tensor")
        total = 0.0
        for t in ts:
            total += t.data
        out = Tensor(total, _op="sum", _prev=set(ts))

        def _backward() -> None:
            for t in ts:
                t.grad += out.grad

        out._backward = _backward
        return out

    @staticmethod
    def mean(tensors: Iterable["Tensor"]) -> "Tensor":
        ts = list(tensors)
        if not ts:
            raise ValueError("mean needs at least one tensor")
        n = len(ts)
        total = 0.0
        for t in ts:
            total += t.data
        out = Tensor(total / n, _op="mean", _prev=set(ts))

        def _backward() -> None:
            for t in ts:
                t.grad += out.grad / n

        out._backward = _backward
        return out

    @staticmethod
    def max(tensors: Iterable["Tensor"]) -> "Tensor":
        """The winning value as a Tensor; the gradient flows to it alone.

        Ties go to the lowest index — deterministic and documented.
        """
        ts = list(tensors)
        if not ts:
            raise ValueError("max needs at least one tensor")
        winner = 0
        best = ts[0].data
        for i in range(1, len(ts)):
            if ts[i].data > best:
                winner, best = i, ts[i].data
        out = Tensor(best, _op="max", _prev=set(ts))

        def _backward() -> None:
            ts[winner].grad += out.grad

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Tensor(data={self.data:.6g}, grad={self.grad:.6g})"
