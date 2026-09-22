import { expect, test, describe } from "bun:test";
const candle = require("../index.node");

const toArr = (t: any) => {
  const { shape, data } = candle.tensorToF32(t);
  return { shape, values: Array.from(new Float32Array(data)) };
};

describe("autograd", () => {
  test("d/dx x^2 at x=3 is 6", () => {
    const x = candle.varFromF32(new Float32Array([3]), [1]);
    const y = candle.tensorMul(x, x);
    const g = candle.gradOf(candle.backward(y), x);
    expect(toArr(g).values[0]).toBeCloseTo(6, 4);
  });

  test("grad of sum(w * x) wrt w is x", () => {
    const w = candle.varFromF32(new Float32Array([1, 1, 1]), [3]);
    const x = candle.tensorFromF32(new Float32Array([2, 5, 7]), [3]);
    const loss = candle.tensorSumAll(candle.tensorMul(w, x));
    const g = candle.gradOf(candle.backward(loss), w);
    expect(toArr(g).values).toEqual([2, 5, 7]);
  });

  test("sgdStep moves params downhill", () => {
    const w = candle.varFromF32(new Float32Array([5]), [1]);
    const loss = candle.tensorMul(w, w); // dL/dw = 2w = 10
    candle.sgdStep(candle.backward(loss), w, 0.1);
    expect(toArr(w).values[0]).toBeCloseTo(4, 4); // 5 - 0.1*10
  });

  test("trains an MLP on XOR", () => {
    const X = candle.tensorFromF32(
      new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]), [4, 2]
    );
    const Y = candle.tensorFromF32(new Float32Array([0, 1, 1, 0]), [4, 1]);

    const w1 = candle.varRandn(0.8, [2, 8]);
    const b1 = candle.varZeros([8]);
    const w2 = candle.varRandn(0.8, [8, 1]);
    const b2 = candle.varZeros([1]);
    const params = [w1, b1, w2, b2];

    const forward = (x: any) => {
      const h = candle.tensorTanh(
        candle.tensorBroadcastAdd(candle.tensorMatmul(x, w1), b1)
      );
      return candle.tensorBroadcastAdd(candle.tensorMatmul(h, w2), b2);
    };

    let loss = 0;
    for (let i = 0; i < 3000; i++) {
      const l = candle.mseLoss(forward(X), Y);
      loss = candle.tensorToScalar(l);
      const grads = candle.backward(l);
      for (const p of params) candle.sgdStep(grads, p, 0.1);
    }

    console.log(`Loss: ${loss.toFixed(3)}`);

    expect(loss).toBeLessThan(0.01);

    const preds = toArr(forward(X)).values.map((v) => (v > 0.5 ? 1 : 0));
    expect(preds).toEqual([0, 1, 1, 0]);
  }, 30000);
});