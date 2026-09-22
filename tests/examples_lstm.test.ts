import { expect, test, describe } from "bun:test";
import candle, { Tensor } from "../lib/index";
import { accuracy, makeRng } from "./_util";

const T = 10;
const B = 32;

describe("example: LSTM", () => {
  test("layer shapes and parameter count", () => {
    const lstm = new candle.nn.LSTM(1, 16);
    expect(lstm.parameters().length).toBe(4); // w_ih, w_hh, b_ih, b_hh
    const out = lstm.forward(candle.zeros([B, T, 1]));
    expect(out.shape).toEqual([B, T, 16]);

    const noBias = new candle.nn.LSTM(3, 8, { bias: false });
    expect(noBias.parameters().length).toBe(2);
    expect(noBias.forward(candle.zeros([2, T, 3])).shape).toEqual([2, T, 8]);
  }, 30000);

  test("learns the running sum of a sequence", () => {
    const rng = makeRng(21);
    const xs: number[] = [];
    const ys: number[] = [];
    for (let b = 0; b < B; b++) {
      let acc = 0;
      for (let t = 0; t < T; t++) {
        const v = rng();
        xs.push(v);
        acc += v;
        ys.push(acc);
      }
    }
    const X = candle.tensor(xs, [B, T, 1]);
    const Y = candle.tensor(ys, [B, T, 1]);

    class SumNet extends candle.nn.Module {
      lstm = this.addModule(new candle.nn.LSTM(1, 16));
      head = this.addModule(new candle.nn.Linear(16, 1));

      protected _forward(x: Tensor): Tensor {
        return this.head.forward(this.lstm.forward(x));
      }
    }

    const model = new SumNet();
    const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.01 });
    let first = 0;
    let loss = 0;
    for (let step = 0; step < 400; step++) {
      const l = candle.F.mseLoss(model.forward(X), Y);
      loss = l.item();
      if (step === 0) first = loss;
      opt.step(l.backward());
    }

    console.log(`  LSTM running-sum MSE: ${first.toFixed(3)} -> ${loss.toFixed(5)}`);
    expect(loss).toBeLessThan(0.02);
    expect(loss).toBeLessThan(first);
  }, 30000);

  test("classifies sequences by their total", () => {
    const rng = makeRng(22);
    const xs: number[] = [];
    const labels: number[] = [];
    for (let b = 0; b < B; b++) {
      let total = 0;
      for (let t = 0; t < T; t++) {
        const v = rng();
        xs.push(v);
        total += v;
      }
      labels.push(total > T / 2 ? 1 : 0);
    }
    const X = candle.tensor(xs, [B, T, 1]);
    const Y = new Uint32Array(labels);

    class SumClassifier extends candle.nn.Module {
      lstm = this.addModule(new candle.nn.LSTM(1, 16));
      drop = this.addModule(new candle.nn.Dropout(0.1));
      head = this.addModule(new candle.nn.Linear(16, 2));

      protected _forward(x: Tensor): Tensor {
        const h = this.lstm.forward(x); // [B, T, 16]
        const last = h.getOnDim(1, T - 1); // [B, 16]
        return this.head.forward(this.drop.forward(last));
      }
    }

    const model = new SumClassifier();
    const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.02 });
    for (let step = 0; step < 300; step++) {
      const l = candle.F.crossEntropy(model.forward(X), Y);
      opt.step(l.backward());
    }

    model.eval(); // disable dropout for evaluation
    const preds = model.forward(X).argmax(1).to("f32").toArray();
    const acc = accuracy(preds, labels);
    console.log(`  LSTM sequence classification acc=${acc}`);
    expect(acc).toBeGreaterThan(0.9);
  }, 30000);
});
