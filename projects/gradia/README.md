# gradia — reverse-mode automatic differentiation from scratch

**Backpropagation from scratch — the engine inside PyTorch, in ~600 lines of pure Python.**
Every operation on a scalar `Tensor` records its parents and a two-line backward closure;
`loss.backward()` walks the graph in reverse topological order and applies the chain rule.
On top of it: a tiny neural-net library (MLP, MSE, softmax cross-entropy, SGD) that
actually trains — a 2→8→1 net fits y = x², a 2→8→8→2 net classifies gaussian blobs.

## Why this matters

`loss.backward()` is the most-called line in deep learning, yet the machinery behind it —
a recorded graph, a topological sort, one local derivative per node — is treated as a
black box by most developers. Building it from scratch proves you know exactly what
happens between the forward pass and the first weight update, because every gradient
rule here is visible, testable, and yours.

## Quickstart

```bash
python -m pytest -q        # 28 tests
python -m gradia demo      # gradient check, x² fit, blob classifier
```

The demo always prints the same numbers: autograd vs finite differences agree to
~1e-11, the regression's MSE falls 4.58 → 0.0076 in 600 epochs, and the classifier
lands at 100% accuracy.

## The ops

Every op builds the graph and installs its own backward rule:

| op | forward | gradient it propagates |
|---|---|---|
| `a + b` | `a + b` | 1 to each parent |
| `a - b` | `a − b` | `+1` to `a`, `−1` to `b` |
| `a * b` | `a · b` | the *other* parent's value |
| `a / b` | `a * b**-1` | composed from `mul` and `pow` — no special case |
| `a ** p` (scalar p) | `aᵖ` | `p · aᵖ⁻¹` |
| `.tanh()` | `tanh(a)` | `1 − tanh²(a)` |
| `.exp()` | `eᵃ` | `eᵃ` (the output itself) |
| `.log()` | `ln(a)` | `1 / a` |
| `.relu()` | `max(0, a)` | 1 above zero, else 0 |
| `.sigmoid()` | `1/(1+e⁻ᵃ)` | `s · (1 − s)` — sign-split, overflow-free |
| `Tensor.sum(xs)` | `Σxs` | 1 to every element (one fused node) |
| `Tensor.mean(xs)` | `Σxs / n` | `1/n` to every element |
| `Tensor.max(xs)` | the winner | 1 to the winning element only (ties → lowest index) |

`backward()` seeds the root with 1.0 and processes the graph newest-first by creation
serial — a topological order by construction, so accumulation order (and therefore the
floats) is identical on every machine.

## API

| call | what it does |
|---|---|
| `Tensor(x)` / `x + y` / `x * 2` … | scalars that remember how they were made; numbers coerce automatically (reflected ops too) |
| `out.backward()` | `d out/d t` lands in every `t.grad` that fed `out` |
| `Module.parameters()` / `.zero_grad()` | every learnable Tensor / clear them all |
| `Neuron(n_in, act)` · `Layer(n_in, n_out, act)` · `MLP(sizes, act)` | fully-connected stack; weights drawn from a fixed-seed LCG, uniform [-1, 1] scaled by 1/√n_in; output layer linear unless `output_activation=` |
| `mse_loss(preds, targets)` | mean of squared diffs — targets can be plain floats |
| `softmax_cross_entropy(logits, class_index)` | `log Σ exp(logits) + peak − logit_y` in engine ops; backprop reproduces `softmax − onehot` on its own |
| `cross_entropy_loss(logits_batch, ys)` / `accuracy(logits, ys)` | batch mean of the above / argmax agreement |
| `SGD(params, lr).step(lr)` | `p -= lr · p.grad`; per-call `lr` overrides |
| `train(model, xs, ys, epochs, lr, loss_fn)` | full-batch loop → `[(epoch, loss), …]` |
| `LCG(seed)` | the deterministic randomness: `.uniform()`, `.normal()` (Box–Muller) |

## Guarantees

- **Gradients are checked, not trusted** — every op is tested against central finite
  differences at two points; the losses are tested against their hand-derived formulas,
  including the classic `softmax − onehot` identity.
- **Deterministic** — a fixed-seed LCG replaces `random`, graph traversal is
  memory-address-independent, and two identical trainings produce bit-identical loss
  histories (tested).
- **It really learns** — the MLP test requires final MSE < 0.05 on x² (it reaches
  0.0076) and ≥ 0.9 accuracy on the blobs (it reaches 1.0).

## Honest scope

Scalars only — no shapes, no broadcasting, no vectorized kernels; a reduction takes a
list of Tensors and fuses it into one node. That constraint is the point: with nowhere
to hide, every rule of reverse-mode autodiff is a two-line closure you can read.
Backward on a leaf or twice without `zero_grad` behaves exactly like the classic
micrograd recipe (seeds/accumulates). Part of an eight-project from-scratch AI systems
portfolio.
