import { expect, test, describe } from "bun:test";
import candle, { Tensor } from "../lib/index";

describe("library: tensors", () => {
  test("creation and readback", () => {
    const a = candle.tensor([[1, 2], [3, 4]]);
    expect(a.shape).toEqual([2, 2]);
    expect(a.tolist()).toEqual([[1, 2], [3, 4]]);
    expect(a.toArray()).toEqual([1, 2, 3, 4]);
    expect(a.numel).toBe(4);
    expect(a.rank).toBe(2);
    expect(a.dtype).toBe("f32");

    expect(candle.zeros([2]).toArray()).toEqual([0, 0]);
    expect(candle.ones([2]).toArray()).toEqual([1, 1]);
    expect(candle.full([2], 7).toArray()).toEqual([7, 7]);
    expect(candle.arange(0, 4).toArray()).toEqual([0, 1, 2, 3]);
    expect(candle.eye(2).tolist()).toEqual([[1, 0], [0, 1]]);
    expect(candle.tril(2).tolist()).toEqual([[1, 0], [1, 1]]);
  });

  test("item() is only for single-element tensors", () => {
    expect(candle.tensor([5]).item()).toBe(5);
    expect(() => candle.tensor([1, 2]).item()).toThrow();
  });

  test("methods chain and broadcast", () => {
    const a = candle.tensor([[1, 2], [3, 4]]);
    const b = candle.tensor([10, 20]);

    expect(a.add(b).tolist()).toEqual([
      [11, 22],
      [13, 24],
    ]);
    expect(a.mul(2).sub(1).relu().tolist()).toEqual([
      [1, 3],
      [5, 7],
    ]);
    expect(a.matmul(candle.eye(2)).tolist()).toEqual([
      [1, 2],
      [3, 4],
    ]);
    expect(a.sum(1).toArray()).toEqual([3, 7]);
    expect(a.mean(0).toArray()).toEqual([2, 3]);
    expect(a.argmax(1).to("f32").toArray()).toEqual([1, 1]);
    expect(a.reshape(4).shape).toEqual([4]);
    expect(a.flatten().shape).toEqual([4]);
    expect(a.t().tolist()).toEqual([
      [1, 3],
      [2, 4],
    ]);
    expect(a.softmax(1).sum(1).toArray()[0]).toBeCloseTo(1, 5);
  });

  test("functional ops and losses", () => {
    const x = candle.tensor([1, 2, 3], [1, 3]);
    expect(candle.F.softmax(x, 1).sum(1).item()).toBeCloseTo(1, 5);

    const logits = candle.tensor([
      [2, 1, 0],
      [0, 1, 2],
    ]);
    expect(candle.F.crossEntropy(logits, new Uint32Array([0, 2])).item())
      .toBeGreaterThan(0);
    expect(candle.F.mseLoss(candle.tensor([1]), candle.tensor([1])).item())
      .toBeCloseTo(0, 6);
  });

  test("autograd", () => {
    const w = candle.variable([3], [1]);
    const g = w.mul(w).backward().get(w);
    expect(g.item()).toBeCloseTo(6, 4);
    expect(w.data.item()).toBeCloseTo(3, 5);
  });
});

describe("library: nn", () => {
  const X = candle.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ]);
  const Y = candle.tensor([[0], [1], [1], [0]]);

  class MLP extends candle.nn.Module {
    fc1 = this.addModule(new candle.nn.Linear(2, 8));
    fc2 = this.addModule(new candle.nn.Linear(8, 1));

    protected _forward(x: Tensor): Tensor {
      return this.fc2.forward(this.fc1.forward(x).tanh());
    }
  }

  test("Sequential MLP trains XOR", () => {
    const model = new candle.nn.Sequential(
      new candle.nn.Linear(2, 8),
      new candle.nn.Tanh(),
      new candle.nn.Linear(8, 1)
    );
    expect(model.parameters().length).toBe(4);

    const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.05 });
    let loss = 0;
    for (let i = 0; i < 500; i++) {
      const l = candle.F.mseLoss(model.forward(X), Y);
      loss = l.item();
      opt.step(l.backward());
    }

    expect(loss).toBeLessThan(0.05);
    expect(model.forward(X).toArray().map((v) => (v > 0.5 ? 1 : 0))).toEqual([
      0, 1, 1, 0,
    ]);
  }, 30000);

  test("custom Module subclass trains", () => {
    const model = new MLP();
    const opt = new candle.optim.SGD(model.parameters(), { lr: 0.1 });
    let loss = 0;
    for (let i = 0; i < 3000; i++) {
      const l = candle.F.mseLoss(model.forward(X), Y);
      loss = l.item();
      opt.step(l.backward());
    }
    expect(loss).toBeLessThan(0.05);
  }, 30000);

  test("save/load round-trips a module", () => {
    const path = "/tmp/candle_lib_model.safetensors";
    const a = new MLP();
    a.save(path);
    const b = new MLP();
    b.load(path);
    expect(b.forward(X).toArray()).toEqual(a.forward(X).toArray());
  });

  test("optimizer learning rate can change", () => {
    const p = candle.variable([5], [1]);
    const opt = new candle.optim.SGD([p], { lr: 0.1 });
    opt.step(p.mul(p).backward());
    expect(p.data.item()).toBeCloseTo(4, 3);

    opt.setLearningRate(0);
    const before = p.data.item();
    opt.step(p.mul(p).backward());
    expect(p.data.item()).toBeCloseTo(before, 5);
  });
});

describe("library: transformer", () => {
  const V = 8;
  const T = 6;
  const D = 16;
  const H = 2;
  const HD = D / H;
  const FF = 4 * D;
  const B = 2;

  const POS = new Uint32Array(Array.from({ length: T }, (_, i) => i));
  const MASK = candle.sub(candle.ones([T, T]), candle.tril(T));

  let seed = 12345;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  const base = Array.from({ length: T + 1 }, () => Math.floor(rnd() * V));
  const X = new Uint32Array([...base.slice(0, T), ...base.slice(0, T)]);
  const Y = new Uint32Array([...base.slice(1), ...base.slice(1)]);

  class MiniGPT extends candle.nn.Module {
    tok = this.addModule(new candle.nn.Embedding(V, D));
    pos = this.addModule(new candle.nn.Embedding(T, D));
    ln1 = this.addModule(new candle.nn.LayerNorm(D));
    ln2 = this.addModule(new candle.nn.LayerNorm(D));
    q = this.addModule(new candle.nn.Linear(D, D));
    k = this.addModule(new candle.nn.Linear(D, D));
    v = this.addModule(new candle.nn.Linear(D, D));
    proj = this.addModule(new candle.nn.Linear(D, D));
    fc1 = this.addModule(new candle.nn.Linear(D, FF));
    fc2 = this.addModule(new candle.nn.Linear(FF, D));
    head = this.addModule(new candle.nn.Linear(D, V));

    protected _forward(ids: Uint32Array): Tensor {
      let h = this.tok.forward(ids, [B, T]).add(this.pos.forward(POS, [T]));

      const n = this.ln1.forward(h);
      const toHeads = (t: Tensor) =>
        t.reshape([B, T, H, HD]).permute([0, 2, 1, 3]);
      const qh = toHeads(this.q.forward(n));
      const kh = toHeads(this.k.forward(n));
      const vh = toHeads(this.v.forward(n));

      const scores = qh.matmul(kh.transpose(2, 3)).affine(1 / Math.sqrt(HD), 0);
      const attn = scores.maskedFill(MASK, -1e9).softmax(3);
      const ctx = attn.matmul(vh).permute([0, 2, 1, 3]).reshape([B, T, D]);
      h = h.add(this.proj.forward(ctx));

      const f = this.fc2.forward(this.fc1.forward(this.ln2.forward(h)).gelu());
      h = h.add(f);

      return this.head.forward(h).reshape([B * T, V]);
    }
  }

  test("MiniGPT overfits a sequence", () => {
    const model = new MiniGPT();
    const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.01 });

    const first = candle.F.crossEntropy(model.forward(X), Y).item();
    let loss = first;
    for (let i = 0; i < 500; i++) {
      const l = candle.F.crossEntropy(model.forward(X), Y);
      loss = l.item();
      opt.step(l.backward());
    }

    console.log(`library transformer loss: ${first.toFixed(3)} -> ${loss.toFixed(4)}`);
    expect(loss).toBeLessThan(0.05);

    const preds = model.forward(X).argmax(1).to("f32").toArray();
    expect(preds).toEqual(Array.from(Y));
  }, 30000);
});
