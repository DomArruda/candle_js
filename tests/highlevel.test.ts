import { expect, test, describe } from "bun:test";
import candle from "../lib/index";
import { gaussian, makeRng } from "./_util";

describe("high-level API: fit", () => {
  test("classification with validation, metrics and callbacks", () => {
    const rng = makeRng(7);
    const perClass = 60;
    const xs: number[] = [];
    const labels: number[] = [];
    const centers = [
      [2, 2],
      [-2, -2],
    ];
    for (let c = 0; c < 2; c++) {
      for (let i = 0; i < perClass; i++) {
        xs.push(centers[c][0] + gaussian(rng), centers[c][1] + gaussian(rng));
        labels.push(c);
      }
    }
    const X = candle.tensor(xs, [perClass * 2, 2]);
    const y = new Uint32Array(labels);

    const model = new candle.nn.Sequential(
      new candle.nn.Linear(2, 16),
      new candle.nn.GELU(),
      new candle.nn.Linear(16, 2)
    );

    let callbacks = 0;
    const history = model.fit(
      { x: X, y },
      {
        epochs: 25,
        batchSize: 24,
        validationSplit: 0.2,
        optimizer: new candle.optim.AdamW(model.parameters(), { lr: 0.05 }),
        metrics: ["accuracy"],
        seed: 3,
        onEpochEnd: (_epoch, logs) => {
          callbacks++;
          expect(typeof logs.valAccuracy).toBe("number");
        },
      }
    );

    expect(history.epochs).toBe(25);
    expect(history.loss.length).toBe(25);
    expect(history.valLoss.length).toBe(25);
    expect(history.metrics.accuracy.length).toBe(25);
    expect(history.valMetrics.accuracy.length).toBe(25);
    expect(callbacks).toBe(25);
    expect(history.valLoss.at(-1)!).toBeLessThan(history.valLoss[0]);
    expect(history.valMetrics.accuracy.at(-1)!).toBeGreaterThan(0.9);

    const evalLogs = model.evaluate({ x: X, y }, { metrics: ["accuracy"] });
    expect(evalLogs.accuracy).toBeGreaterThan(0.9);
    expect(candle.predictClasses(model, X).length).toBe(perClass * 2);
    expect(model.numParameters).toBeGreaterThan(0);
  }, 30000);

  test("regression auto-selects MSE and works without validation", () => {
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
      new candle.nn.Linear(1, 16),
      new candle.nn.Tanh(),
      new candle.nn.Linear(16, 1)
    );
    const history = model.fit(
      { x: X, y: Y },
      {
        epochs: 80,
        batchSize: 16,
        optimizer: new candle.optim.AdamW(model.parameters(), { lr: 0.02 }),
        seed: 5,
      }
    );

    expect(history.valLoss.length).toBe(0);
    expect(history.loss.at(-1)!).toBeLessThan(history.loss[0]);
    expect(history.metrics.mse.at(-1)!).toBeLessThan(0.1);

    const pred = model.predict(X);
    expect(pred.shape).toEqual([n, 1]);
    expect(pred.toArray().length).toBe(n);
  }, 30000);
});

describe("high-level API: data helpers", () => {
  test("trainTestSplit keeps x and y in sync", () => {
    const X = candle.tensor(Array.from({ length: 120 }, (_, i) => i), [120, 1]);
    const y = new Uint32Array(Array.from({ length: 120 }, (_, i) => i % 2));

    const { train, test } = candle.trainTestSplit(
      { x: X, y },
      { testSize: 0.25, seed: 1 }
    );

    expect(train.x.shape[0]).toBe(90);
    expect(test.x.shape[0]).toBe(30);
    expect((train.y as Uint32Array).length).toBe(90);
    expect((test.y as Uint32Array).length).toBe(30);

    const trainXs = train.x.toArray();
    const trainYs = Array.from(train.y as Uint32Array);
    for (let i = 0; i < trainXs.length; i++) {
      expect(trainYs[i]).toBe(trainXs[i] % 2);
    }
  });

  test("top-level helpers and namespaces are wired up", () => {
    expect(typeof candle.fit).toBe("function");
    expect(typeof candle.evaluate).toBe("function");
    expect(typeof candle.trainTestSplit).toBe("function");
    expect(candle.hl.fit).toBe(candle.fit);
    expect(typeof candle.nn.Linear.prototype.fit).toBe("function");
    expect(typeof candle.nn.Linear.prototype.predict).toBe("function");
  });
});
