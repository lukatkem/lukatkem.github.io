"""Hydro-1 tests — the ones that matter for a from-scratch model.

1. Shape/forward sanity
2. Causal mask correctness: position t must never attend to t+1…
3. Tokenizer roundtrip
4. THE real test: the model can overfit a tiny corpus (loss collapses) —
   proof the training loop, gradients, and optimizer wiring all work.
5. Academy: .env key roundtrip, live config propagation, job log parsing.
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydro1 import academy  # noqa: E402
from hydro1.config import all_teachers  # noqa: E402
from hydro1.model import GPT  # noqa: E402
from hydro1.tokenizer import CharTokenizer  # noqa: E402

TINY = dict(vocab_size=20, d_model=32, n_layers=2, n_heads=2, block_size=16)


def test_forward_shapes_and_loss():
    m = GPT(**TINY)
    x = torch.randint(0, 20, (2, 16))
    logits, loss = m(x, targets=x)
    assert logits.shape == (2, 16, 20)
    assert loss.item() > 0
    assert 4000 < m.num_params(non_embedding=False) < 200_000


def test_causal_mask_attention_cannot_see_future():
    m = GPT(**TINY)
    m.eval()
    x = torch.randint(0, 20, (1, 8))
    m(x)
    att = m.blocks[0].attn.attn_map[0]  # (heads, T, T) — block.attn is the module
    future = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    assert (att[:, future] == 0).all(), "attention leaked into the future"


def test_generation_is_autoregressive_and_stable():
    torch.manual_seed(0)
    m = GPT(**TINY)
    m.eval()
    idx = torch.tensor([[1, 2, 3]])
    out = m.generate(idx, max_new_tokens=10, temperature=0.9, top_k=5)
    assert out.shape == (1, 13)
    assert (out[:, :3] == idx).all(), "prompt tokens changed during generation"


def test_tokenizer_roundtrip():
    tok = CharTokenizer.from_text("hello world! 123\n")
    ids = tok.encode("hello world")
    assert tok.decode(ids) == "hello world"
    assert tok.vocab_size == len(set("hello world! 123\n"))
    assert tok.encode("café ☕")[0] == tok.stoi[" "]  # unseen chars → space, no crash


def test_model_can_overfit_tiny_corpus():
    """The loop learns: 200 steps on 400 chars must drive loss to near zero."""
    torch.manual_seed(42)
    text = "abcabcabc" * 44  # trivially predictable
    tok = CharTokenizer.from_text(text)
    data = torch.tensor([tok.encode(text)], dtype=torch.long).squeeze(0)
    m = GPT(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=2, block_size=32)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(200):
        ix = torch.randint(0, len(data) - 33, (8,))
        x = torch.stack([data[i : i + 32] for i in ix])
        y = torch.stack([data[i + 1 : i + 33] for i in ix])
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.25, f"model failed to overfit tiny corpus (loss {loss.item():.3f})"


# ---------- academy ----------

def test_academy_env_roundtrip(tmp_path):
    env = tmp_path / ".env"
    saved_as = academy.save_key("sk-or-v1-abc123", path=env)
    assert saved_as == "OPENROUTER_API_KEY"  # prefix routes the key to the right var
    assert academy.load_env(env)["OPENROUTER_API_KEY"] == "sk-or-v1-abc123"

    academy.save_key("nvapi-xyz", path=env)
    values = academy.load_env(env)
    assert values["NVIDIA_API_KEY"] == "nvapi-xyz"
    assert values["OPENROUTER_API_KEY"] == "sk-or-v1-abc123"  # first key untouched

    # apply_env pushes saved keys into os.environ — that's how keys saved
    # from the UI reach config/teachers without a server restart
    old = {k: os.environ.get(k) for k in ("OPENROUTER_API_KEY", "NVIDIA_API_KEY")}
    try:
        for k in old:
            os.environ.pop(k, None)
        academy.apply_env(env)
        assert os.environ.get("OPENROUTER_API_KEY") == "sk-or-v1-abc123"
        cloud = [t for t in all_teachers() if t["tier"] != "local"]
        assert cloud, "cloud teachers must appear once a key is set"
        # each cloud teacher carries its own routing
        assert all("base_url" in t and "key_env" in t for t in cloud)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_academy_clear_key(tmp_path):
    env = tmp_path / ".env"
    academy.save_key("sk-temp", path=env)
    academy.save_key("nvapi-temp", path=env)
    academy.clear_key(env)
    values = academy.load_env(env)
    assert "TEACHER_API_KEY" not in values
    assert "OPENROUTER_API_KEY" not in values
    assert "NVIDIA_API_KEY" not in values


def test_academy_job_progress_parsing():
    # parse without spawning a real subprocess
    j = academy.Job.__new__(academy.Job)
    j.kind, j.stage, j.done, j.total = "distill", "", 0, 0
    j._parse("  generated 40/200 → kept 39")
    assert (j.stage, j.done, j.total) == ("writing stories", 40, 200)
    j._parse("grading: 195 → 180 passed")
    assert (j.stage, j.done, j.total) == ("grading", 180, 195)
    j._parse("wrote 180 stories · 104 KB → …")
    assert j.stage == "done"

    t = academy.Job.__new__(academy.Job)
    t.kind, t.stage, t.done, t.total = "train", "", 0, 8000
    t._parse("step  3000 · loss 0.1234 · val 0.5678 · 100s")
    assert (t.stage, t.done) == ("training", 3000)
    t._parse("done in 152.3 min · best val 0.5012 → …")
    assert t.stage == "done" and t.done == 8000


def test_distill_verdict_parsing():
    from hydro1.distill import parse_verdict

    assert parse_verdict("VERDICT: PASS")
    assert not parse_verdict("VERDICT: FAIL")
    # reasoning models deliberate first, verdict comes last
    assert parse_verdict("Let me check coherence… it holds up. VERDICT: PASS")
    assert not parse_verdict("Strong candidate, but the ending drifts. VERDICT: FAIL")
    assert not parse_verdict("does not pass")  # no explicit marker → reject
    assert not parse_verdict("")  # budget exhausted mid-reasoning → reject
