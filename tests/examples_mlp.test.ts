import { expect, test, describe } from "bun:test";
import candle, { Tensor } from "../lib/index";
import { accuracy, flatten, gaussian, makeRng } from "./_util";

describe("example: MLP regression (fit sin)", () => {
  test("learns a smooth 1-D function", () => {
    const n = 64;
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = -3 + (6 * i) / (n - 1);
      xs.push(x);
      ys.push(Math.sin(x));
    }
    const X = candle.tensor(xs, [n, 1]);
    const Y = candle.tensor(ys, [n, 1]);

    const model = new candle.nn.Sequential(
      new candle.nn.Linear(1, 32),
      new candle.nn.Tanh(),
      new candle.nn.Linear(32, 1)
    );
    const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.01 });

    let first = 0;
    let loss = 0;
    for (let step = 0; step < 600; step++) {
      const l = candle.F.mseLoss(model.forward(X), Y);
      loss = l.item();
      if (step === 0) first = loss;
      opt.step(l.backward());
    }

    console.log(`  sin fit MSE: ${first.toFixed(4)} -> ${loss.toFixed(5)}`);
    expect(loss).toBeLessThan(0.01);
    expect(loss).toBeLessThan(first);
  }, 30000);
});

describe("example: MLP classification (two blobs)", () => {
  test("separates two Gaussian clusters", () => {
    const rng = makeRng(7);
    const perClass = 100;
    const xs: number[] = [];
    const labels: number[] = [];
    const centers = [
      [2, 2],
      [-2, -2],
    ];
    for (let c = 0; c < 2; c++) {
      for (let i = 0; i < perClass; i++) {
        xs.push(centers[c][0] + gaussian(rng));
        xs.push(centers[c][1] + gaussian(rng));
        labels.push(c);
      }
    }
    const X = candle.tensor(xs, [perClass * 2, 2]);
    const Y = new Uint32Array(labels);

    const model = new candle.nn.Sequential(
      new candle.nn.Linear(2, 16),
      new candle.nn.GELU(),
      new candle.nn.Linear(16, 2)
    );
    const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.05 });

    let loss = 0;
    for (let step = 0; step < 300; step++) {
      const l = candle.F.crossEntropy(model.forward(X), Y);
      loss = l.item();
      opt.step(l.backward());
    }

    const preds = model.forward(X).argmax(1).to("f32").toArray();
    const acc = accuracy(preds, labels);
    console.log(`  blob classification: loss=${loss.toFixed(4)} acc=${acc}`);
    expect(loss).toBeLessThan(0.1);
    expect(acc).toBeGreaterThan(0.95);
  }, 30000);
});

describe("example: activation zoo", () => {
  test("every activation keeps the expected shape and values", () => {
    const x = candle.tensor([-2, -0.5, 0, 0.5, 2]);

    expect(x.relu().toArray()).toEqual([0, 0, 0, 0.5, 2]);
    expect(x.relu2().toArray()).toEqual([0, 0, 0, 0.25, 4]);
    expect(x.relu6().toArray()).toEqual([0, 0, 0, 0.5, 2]);
    expect(x.sigmoid().toArray()[2]).toBeCloseTo(0.5, 5);
    expect(x.tanh().toArray()[2]).toBeCloseTo(0, 5);
    expect(x.silu().toArray()[2]).toBeCloseTo(0, 5);
    expect(x.gelu().toArray()[2]).toBeCloseTo(0, 5);
    expect(x.geluErf().toArray()[2]).toBeCloseTo(0, 5);
    expect(x.mish().toArray()[2]).toBeCloseTo(0, 5);
    expect(x.erf().toArray()[4]).toBeCloseTo(0.9953, 3);
    expect(x.hardSigmoid().toArray()[2]).toBeCloseTo(0.5, 5);
    expect(x.selu().toArray()[4]).toBeCloseTo(2 * 1.0507, 3);

    // SwiGLU halves the last dimension: [4] -> [2].
    const g = candle.tensor([1, 2, 3, 4]);
    expect(g.swiGLU().shape).toEqual([2]);
    const silu = (v: number) => v / (1 + Math.exp(-v));
    const swiglu = g.swiGLU().toArray();
    expect(swiglu[0]).toBeCloseTo(silu(1) * 3, 5);
    expect(swiglu[1]).toBeCloseTo(silu(2) * 4, 5);

    // Functional forms match the methods.
    expect(candle.F.swiGLU(g).toArray()).toEqual(g.swiGLU().toArray());
    expect(candle.F.erf(x).toArray()).toEqual(x.erf().toArray());
    expect(candle.F.hardSwish(x).toArray()).toEqual(x.hardSwish().toArray());
  });
});

describe("example: classification with a custom Module", () => {
  test("a 3-layer classifier reaches high accuracy", () => {
    const rng = makeRng(99);
    const n = 150;
    const xs: number[] = [];
    const labels: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = gaussian(rng);
      const y = gaussian(rng);
      xs.push(x, y);
      labels.push(x * y > 0 ? 1 : 0); // XOR-like quadrants
    }
    const X = candle.tensor(xs, [n, 2]);
    const Y = new Uint32Array(labels);

    class Classifier extends candle.nn.Module {
      l1 = this.addModule(new candle.nn.Linear(2, 16));
      l2 = this.addModule(new candle.nn.Linear(16, 16));
      l3 = this.addModule(new candle.nn.Linear(16, 2));

      protected _forward(x: Tensor): Tensor {
        const h = this.l1.forward(x).tanh();
        return this.l3.forward(this.l2.forward(h).tanh());
      }
    }

    const model = new Classifier();
    const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.02 });
    for (let step = 0; step < 400; step++) {
      const l = candle.F.crossEntropy(model.forward(X), Y);
      opt.step(l.backward());
    }

    const preds = model.forward(X).argmax(1).to("f32").toArray();
    const acc = accuracy(preds, labels);
    console.log(`  quadrant classification acc=${acc}`);
    expect(acc).toBeGreaterThan(0.9);
    expect(flatten(preds).length).toBe(n);
  }, 30000);
});
