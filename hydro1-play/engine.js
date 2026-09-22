/* Hydro-1 browser engine — a decoder-only GPT in plain JavaScript.
   The weights (fp16 → fp32) were exported from checkpoints trained from
   scratch on a laptop; this file computes attention, MLPs and sampling on
   the VISITOR'S machine. No server, no API, no data leaves the browser.

   Mirrors hydro1/model.py and hydro1/verify_export.py line for line:
   pre-LN blocks (ln1 → causal MHA → residual, ln2 → GELU MLP → residual),
   final LayerNorm, weight-tied head. */
"use strict";

function f16to32(u) {
  const s = (u & 0x8000) >> 15, e = (u & 0x7c00) >> 10, m = u & 0x03ff;
  let v;
  if (e === 0) v = m * Math.pow(2, -24);
  else if (e === 31) v = m ? NaN : Infinity;
  else v = Math.pow(2, e - 15) * (1 + m / 1024);
  return s ? -v : v;
}

export class HydroEngine {
  constructor(manifest, binBuffer, onProgress) {
    const cfg = manifest.config;
    this.cfg = cfg;
    this.chars = manifest.chars;               // index = token id
    this.stoi = Object.fromEntries(this.chars.map((c, i) => [c, i]));
    this.d = cfg.d_model; this.L = cfg.n_layers; this.H = cfg.n_heads;
    this.dh = cfg.d_model / cfg.n_heads; this.Tmax = cfg.block_size; this.V = cfg.vocab_size;

    // parse the blob: [u16 nameLen][name][u8 ndim][ndim × i32 dims][f16 data]
    const dv = new DataView(binBuffer);
    const u8 = new Uint8Array(binBuffer);
    let off = 0; const raw = {};
    while (off < binBuffer.byteLength) {
      const nl = dv.getUint16(off, true); off += 2;
      const name = new TextDecoder().decode(u8.subarray(off, off + nl)); off += nl;
      const ndim = dv.getUint8(off); off += 1;
      const dims = []; for (let i = 0; i < ndim; i++) { dims.push(dv.getInt32(off, true)); off += 4; }
      const n = dims.reduce((a, b) => a * b, 1);
      const f32 = new Float32Array(n);
      for (let i = 0; i < n; i++) f32[i] = f16to32(dv.getUint16(off + 2 * i, true));
      off += 2 * n;
      raw[name] = { f32, dims };
      if (onProgress) onProgress(off / binBuffer.byteLength);
    }
    this.w = {}; for (const k in raw) this.w[k] = raw[k].f32;
    this.kv = Array.from({ length: this.L }, () => ({
      k: new Float32Array(this.Tmax * this.H * this.dh),
      v: new Float32Array(this.Tmax * this.H * this.dh),
    }));
    this.pos = 0; this.ids = [];
  }

  reset() { this.pos = 0; this.ids = []; }

  ln(x, out, w, b) {
    let m = 0; for (let i = 0; i < this.d; i++) m += x[i]; m /= this.d;
    let v = 0; for (let i = 0; i < this.d; i++) { const d = x[i] - m; v += d * d; } v /= this.d;
    const inv = 1 / Math.sqrt(v + 1e-5);
    for (let i = 0; i < this.d; i++) out[i] = (x[i] - m) * inv * w[i] + b[i];
  }

  // out[j] = Σ_i x[i] * W[j*inN + i]  (weights stored [out, in], as nn.Linear)
  linear(x, W, out, outN, inN) {
    for (let j = 0; j < outN; j++) {
      let s = 0; const row = j * inN;
      for (let i = 0; i < inN; i++) s += x[i] * W[row + i];
      out[j] = s;
    }
  }

  forward(tok) {  // returns the next-token distribution over the vocabulary
    const { d, H, dh } = this;
    if (this.pos >= this.Tmax) {   // slide the window: restart from the tail
      const tail = this.ids.slice(-(this.Tmax - 1));
      this.reset();
      for (const t of tail) this.forward(t);
    }
    const x = new Float32Array(d);
    for (let i = 0; i < d; i++) x[i] = this.w["tok_emb.weight"][tok * d + i] + this.w["pos_emb.weight"][this.pos * d + i];
    const p = this.pos;
    this.ids.push(tok); this.pos++;

    const h = new Float32Array(d), qkv = new Float32Array(3 * d),
          tmp = new Float32Array(4 * d), att = new Float32Array(p + 1);
    for (let l = 0; l < this.L; l++) {
      const b = `blocks.${l}.`;
      this.ln(x, h, this.w[b + "ln1.weight"], this.w[b + "ln1.bias"]);
      this.linear(h, this.w[b + "attn.qkv.weight"], qkv, 3 * d, d);
      // qkv rows: [0..d) = Q · [d..2d) = K · [2d..3d) = V — split per head
      for (let hh = 0; hh < H; hh++) {
        for (let j = 0; j < dh; j++) {
          this.kv[l].k[(p * H + hh) * dh + j] = qkv[d + hh * dh + j];
          this.kv[l].v[(p * H + hh) * dh + j] = qkv[2 * d + hh * dh + j];
        }
      }
      // causal attention — softmax is PER HEAD (each head has its own
      // distribution over positions 0..p), then heads are concatenated
      for (let hh = 0; hh < H; hh++) {
        let maxs = -Infinity;
        for (let pp = 0; pp <= p; pp++) {
          let s = 0;
          for (let j = 0; j < dh; j++)
            s += qkv[hh * dh + j] * this.kv[l].k[(pp * H + hh) * dh + j];
          s /= Math.sqrt(dh);
          att[pp] = s; if (s > maxs) maxs = s;
        }
        let sum = 0;
        for (let pp = 0; pp <= p; pp++) { att[pp] = Math.exp(att[pp] - maxs); sum += att[pp]; }
        for (let j = 0; j < dh; j++) {
          let y = 0;
          for (let pp = 0; pp <= p; pp++) y += att[pp] * this.kv[l].v[(pp * H + hh) * dh + j];
          h[hh * dh + j] = y / sum;
        }
      }
      this.linear(h, this.w[b + "attn.proj.weight"], qkv, d, d);
      for (let i = 0; i < d; i++) x[i] += qkv[i];
      this.ln(x, h, this.w[b + "ln2.weight"], this.w[b + "ln2.bias"]);
      this.linear(h, this.w[b + "mlp.fc.weight"], tmp, 4 * d, d);
      for (let j = 0; j < 4 * d; j++) {
        const u = tmp[j];
        tmp[j] = 0.5 * u * (1 + Math.tanh(0.7978845608 * (u + 0.044715 * u * u * u)));
      }
      this.linear(tmp, this.w[b + "mlp.proj.weight"], h, d, 4 * d);
      for (let i = 0; i < d; i++) x[i] += h[i] + this.w[b + "mlp.proj.bias"][i];
    }
    const xf = new Float32Array(d);
    this.ln(x, xf, this.w["ln_f.weight"], this.w["ln_f.bias"]);
    const logits = new Float32Array(this.V);
    for (let v = 0; v < this.V; v++) {
      let s = 0; const row = v * d;
      for (let i = 0; i < d; i++) s += xf[i] * this.w["tok_emb.weight"][row + i];
      logits[v] = s;
    }
    return logits;
  }

  sample(logits, temperature = 0.8, topK = 40) {
    const t = Math.max(temperature, 1e-6);
    const idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, topK);
    const mx = logits[idx[0]] / t;
    const probs = idx.map(i => Math.exp(logits[i] / t - mx));
    const sum = probs.reduce((a, b) => a + b, 0);
    let r = Math.random() * sum;
    for (let i = 0; i < idx.length; i++) { r -= probs[i]; if (r <= 0) return idx[i]; }
    return idx[0];
  }
}

// Warm the context with all but the last prompt character, then generate `n`
// tokens, streaming one callback per token. The last character is held back —
// it becomes the first input of the generation loop (never feed it twice).
export async function generate(engine, prompt, n, onChar, temperature = 0.8) {
  engine.reset();
  const cs = [...prompt];
  let tok = 0;
  for (let i = 0; i < cs.length; i++) {
    tok = engine.stoi[cs[i]] ?? 0;
    if (i < cs.length - 1) {
      engine.forward(tok);
      await new Promise(r => setTimeout(r, 0));   // keep the tab responsive
    }
  }
  for (let i = 0; i < n; i++) {
    const logits = engine.forward(tok);
    tok = engine.sample(logits, temperature);
    onChar(engine.chars[tok] ?? "");
    if (i % 4 === 3) await new Promise(r => setTimeout(r, 0));
  }
}
