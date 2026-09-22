import { expect, test, describe } from "bun:test";
import candle, { Tensor } from "../lib/index";
import { accuracy, makeRng } from "./_util";

const V = 6;
const T = 8;
const B = 32;
const D = 16;
const H = 2;
const HD = D / H;
const FF = 4 * D;
const POS = new Uint32Array(Array.from({ length: T }, (_, i) => i));

/** A single bidirectional transformer block (post-norm residual). */
class Block extends candle.nn.Module {
  ln1 = this.addModule(new candle.nn.LayerNorm(D));
  q = this.addModule(new candle.nn.Linear(D, D));
  k = this.addModule(new candle.nn.Linear(D, D));
  v = this.addModule(new candle.nn.Linear(D, D));
  proj = this.addModule(new candle.nn.Linear(D, D));
  ln2 = this.addModule(new candle.nn.LayerNorm(D));
  fc1 = this.addModule(new candle.nn.Linear(D, FF));
  fc2 = this.addModule(new candle.nn.Linear(FF, D));

  private toHeads(t: Tensor, batch: number, seq: number): Tensor {
    return t.reshape([batch, seq, H, HD]).permute([0, 2, 1, 3]);
  }

  protected _forward(x: Tensor): Tensor {
    const [batch, seq] = x.shape;
    const n = this.ln1.forward(x);
    const qh = this.toHeads(this.q.forward(n), batch, seq);
    const kh = this.toHeads(this.k.forward(n), batch, seq);
    const vh = this.toHeads(this.v.forward(n), batch, seq);

    const scores = qh.matmul(kh.transpose(2, 3)).affine(1 / Math.sqrt(HD), 0);
    const ctx = scores
      .softmax(3)
      .matmul(vh)
      .permute([0, 2, 1, 3])
      .reshape([batch, seq, D]);
    const h = x.add(this.proj.forward(ctx));

    const f = this.fc2.forward(this.fc1.forward(this.ln2.forward(h)).gelu());
    return h.add(f);
  }
}

/** Embedding + positional + two blocks + mean-pool classification head. */
class Encoder extends candle.nn.Module {
  tok = this.addModule(new candle.nn.Embedding(V, D));
  pos = this.addModule(new candle.nn.Embedding(T, D));
  b0 = this.addModule(new Block());
  b1 = this.addModule(new Block());
  head = this.addModule(new candle.nn.Linear(D, 2));

  protected _forward(ids: Uint32Array): Tensor {
    const batch = ids.length / T;
    let h = this.tok
      .forward(ids, [batch, T])
      .add(this.pos.forward(POS, [T]));
    h = this.b0.forward(h);
    h = this.b1.forward(h);
    return this.head.forward(h.mean(1)); // [B, 2]
  }
}

describe("example: transformer encoder for classification", () => {
  const rng = makeRng(31);
  const ids: number[] = [];
  const labels: number[] = [];
  const TARGET = 1;
  for (let b = 0; b < B; b++) {
    let seen = 0;
    for (let t = 0; t < T; t++) {
      const id = Math.floor(rng() * V);
      ids.push(id);
      if (id === TARGET) seen = 1;
    }
    labels.push(seen);
  }
  const X = new Uint32Array(ids);
  const Y = new Uint32Array(labels);

  test("detects whether a token appears in the sequence", () => {
    const model = new Encoder();
    const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.01 });

    let loss = 0;
    for (let step = 0; step < 300; step++) {
      const l = candle.F.crossEntropy(model.forward(X), Y);
      loss = l.item();
      opt.step(l.backward());
    }

    const preds = model.forward(X).argmax(1).to("f32").toArray();
    const acc = accuracy(preds, labels);
    console.log(`  transformer classification: loss=${loss.toFixed(4)} acc=${acc}`);
    expect(loss).toBeLessThan(0.2);
    expect(acc).toBeGreaterThan(0.9);
  }, 30000);

  test("attention output is finite and shaped [B, T, D]", () => {
    const block = new Block();
    const x = candle.randn([B, T, D]);
    const y = block.forward(x);
    expect(y.shape).toEqual([B, T, D]);
    expect(y.toArray().every((v) => Number.isFinite(v))).toBe(true);
  }, 30000);
});
