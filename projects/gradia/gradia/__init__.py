"""gradia — reverse-mode automatic differentiation from scratch.

The engine inside PyTorch, in pure python: a scalar Tensor that records every
operation, backpropagates by walking the graph in reverse topological order,
and a tiny neural-network library — layers, losses, SGD — built on top of it.
No torch, no numpy, no random module: pure stdlib, fully deterministic.
"""
from .engine import Tensor
from .nn import MLP, LCG, Module, Neuron, SGD, Layer, accuracy, cross_entropy_loss, mse_loss, softmax_cross_entropy
from .train import train

__all__ = [
    "Tensor",
    "Module",
    "Neuron",
    "Layer",
    "MLP",
    "SGD",
    "LCG",
    "mse_loss",
    "softmax_cross_entropy",
    "cross_entropy_loss",
    "accuracy",
    "train",
]
__version__ = "1.0.0"
