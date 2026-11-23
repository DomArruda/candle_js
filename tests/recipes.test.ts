import { expect, test, describe } from "bun:test";
import candle, { Tensor } from "../lib/index";
import { makeRng } from "./_util";

describe("recipe: raw parameters", () => {
  test("linear regression with variables and functional ops", () => {
    const rng = makeRng(5);
    const n = 128;
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = -2 + 4 * rng();
      xs.push(x);
      ys.push(2 * x + 1);
    }
    const X = candle.tensor(xs, [n, 1]);
    const Y = candle.tensor(ys, [n, 1]);

    const w = candle.variable([0], [1, 1]);
    const b = candle.variable([0], [1]);
    const opt = new candle.optim.AdamW([w, b], { lr: 0.1 });

    const grads = candle.F.mseLoss(candle.F.linear(X, w, b), Y).backward();
    expect(grads.get(w).shape).toEqual([1, 1]);
    expect(grads.get(b).shape).toEqual([1]);
    opt.step(grads);

    let loss = 0;
    for (let step = 1; step < 300; step++) {
      const l = candle.F.mseLoss(candle.F.linear(X, w, b), Y);
      loss = l.item();
      opt.step(l.backward());
    }

    console.log(`  raw-parameter regression: loss=${loss.toFixed(5)}`);
    expect(loss).toBeLessThan(0.01);
    expect(w.item()).toBeCloseTo(2, 1);
    expect(b.item()).toBeCloseTo(1, 1);
  }, 30000);
});

describe("recipe: one trunk, two heads", () => {
  test("trains two tasks through a shared encoder", () => {
    const rng = makeRng(11);
    const n = 128;
    const xs: number[] = [];
    const targets: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = -1 + 2 * rng();
      const y = -1 + 2 * rng();
      xs.push(x, y);
      targets.push(x + y, x - y);
    }
    const X = candle.tensor(xs, [n, 2]);
    const Y = candle.tensor(targets, [n, 2]);

    class MultiTask extends candle.nn.Module {
      trunk = this.addModule(
        new candle.nn.Sequential(
          new candle.nn.Linear(2, 16),
          new candle.nn.Tanh()
        )
      );
      sum = this.addModule(new candle.nn.Linear(16, 1));
      diff = this.addModule(new candle.nn.Linear(16, 1));

      protected _forward(x: Tensor): Tensor {
        const h = this.trunk.forward(x);
        return candle.cat([this.sum.forward(h), this.diff.forward(h)], 1);
      }
    }

    const model = new MultiTask();
    const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.02 });
    const first = candle.F.mseLoss(model.forward(X), Y).item();

    let loss = first;
    for (let step = 0; step < 400; step++) {
      const l = candle.F.mseLoss(model.forward(X), Y);
      loss = l.item();
      opt.step(l.backward());
    }

    console.log(`  shared-trunk MSE: ${first.toFixed(4)} -> ${loss.toFixed(4)}`);
    expect(loss).toBeLessThan(0.01);
    expect(loss).toBeLessThan(first);
  }, 30000);
});

describe("recipe: save and resume", () => {
  test("reloads weights and keeps improving", () => {
    const path = "/tmp/candle_recipe_resume.safetensors";
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

    const make = () =>
      new candle.nn.Sequential(
        new candle.nn.Linear(1, 16),
        new candle.nn.Tanh(),
        new candle.nn.Linear(16, 1)
      );

    const a = make();
    const optA = new candle.optim.AdamW(a.parameters(), { lr: 0.02 });
    for (let step = 0; step < 20; step++) {
      optA.step(candle.F.mseLoss(a.forward(X), Y).backward());
    }
    const lossA = candle.F.mseLoss(a.forward(X), Y).item();
    a.save(path);

    const b = make();
    b.load(path);
    expect(b.forward(X).toArray()).toEqual(a.forward(X).toArray());

    const optB = new candle.optim.AdamW(b.parameters(), { lr: 0.02 });
    let lossB = lossA;
    for (let step = 0; step < 200; step++) {
      const l = candle.F.mseLoss(b.forward(X), Y);
      lossB = l.item();
      optB.step(l.backward());
    }

    console.log(`  resume MSE: ${lossA.toFixed(4)} -> ${lossB.toFixed(5)}`);
    expect(lossB).toBeLessThan(lossA);
  }, 30000);
});
