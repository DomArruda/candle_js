import { expect, test, describe } from "bun:test";
const candle = require("../index.node");

const toArr = (t: any) => {
  const { shape, data } = candle.tensorToF32(t);
  return { shape, values: Array.from(new Float32Array(data)) };
};

const adamw = (vm: any, lr = 0.01) =>
  candle.adamwNew(vm, lr, 0.9, 0.999, 1e-8, 0.0);

describe("optimizers", () => {
  test("adamw moves params downhill", () => {
    const vm = candle.varmapNew();
    const fc = candle.linearNew(vm, "fc", 1, 1, true);
    const opt = adamw(vm, 0.1);

    const x = candle.tensorFromF32(new Float32Array([1]), [1, 1]);
    const y = candle.tensorFromF32(new Float32Array([5]), [1, 1]);

    const before = candle.tensorToScalar(
      candle.mseLoss(candle.linearForward(fc, x), y)
    );
    for (let i = 0; i < 20; i++) {
      const l = candle.mseLoss(candle.linearForward(fc, x), y);
      candle.adamwStep(opt, candle.backward(l));
    }
    const after = candle.tensorToScalar(
      candle.mseLoss(candle.linearForward(fc, x), y)
    );

    expect(after).toBeLessThan(before);
  });

  test("adamw trains XOR faster than raw SGD", () => {
    const vm = candle.varmapNew();
    const fc1 = candle.linearNew(vm, "fc1", 2, 8, true);
    const fc2 = candle.linearNew(vm, "fc2", 8, 1, true);
    const opt = adamw(vm, 0.05);

    const X = candle.tensorFromF32(
      new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]), [4, 2]
    );
    const Y = candle.tensorFromF32(new Float32Array([0, 1, 1, 0]), [4, 1]);

    const forward = (x: any) =>
      candle.linearForward(fc2, candle.tensorTanh(candle.linearForward(fc1, x)));

    let loss = 0;
    for (let i = 0; i < 400; i++) {
      const l = candle.mseLoss(forward(X), Y);
      loss = candle.tensorToScalar(l);
      candle.adamwStep(opt, candle.backward(l));
    }

    expect(loss).toBeLessThan(0.01); // 400 steps vs SGD's 3000
    expect(toArr(forward(X)).values.map((v) => (v > 0.5 ? 1 : 0)))
      .toEqual([0, 1, 1, 0]);
  }, 30000);

  test("sgd optimizer object works too", () => {
    const vm = candle.varmapNew();
    const fc1 = candle.linearNew(vm, "fc1", 2, 8, true);
    const fc2 = candle.linearNew(vm, "fc2", 8, 1, true);
    const opt = candle.sgdNew(vm, 0.1);

    const X = candle.tensorFromF32(
      new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]), [4, 2]
    );
    const Y = candle.tensorFromF32(new Float32Array([0, 1, 1, 0]), [4, 1]);

    const forward = (x: any) =>
      candle.linearForward(fc2, candle.tensorTanh(candle.linearForward(fc1, x)));

    let loss = 0;
    for (let i = 0; i < 3000; i++) {
      const l = candle.mseLoss(forward(X), Y);
      loss = candle.tensorToScalar(l);
      candle.sgdOptStep(opt, candle.backward(l));
    }

    expect(loss).toBeLessThan(0.01);
  }, 30000);

  test("setLr changes the step size", () => {
    const vm = candle.varmapNew();
    candle.linearNew(vm, "fc", 1, 1, false);
    const opt = candle.sgdNew(vm, 0.1);
    candle.sgdSetLr(opt, 0.0); // freeze

    const x = candle.tensorFromF32(new Float32Array([1]), [1, 1]);
    const y = candle.tensorFromF32(new Float32Array([5]), [1, 1]);
    const fc = candle.linearNew(vm, "fc2", 1, 1, false);

    const l = candle.mseLoss(candle.linearForward(fc, x), y);
    const before = candle.tensorToScalar(l);
    candle.sgdOptStep(opt, candle.backward(l));
    const after = candle.tensorToScalar(
      candle.mseLoss(candle.linearForward(fc, x), y)
    );

    expect(after).toBe(before); // lr=0 -> no movement
  });

  test("weight decay shrinks weights toward zero", () => {
    const vm = candle.varmapNew();
    const fc = candle.linearNew(vm, "fc", 4, 4, false);
    const opt = candle.adamwNew(vm, 0.1, 0.9, 0.999, 1e-8, 0.5);

    const x = candle.tensorFromF32(new Float32Array(4).fill(1), [1, 4]);
    const y = candle.tensorFromF32(new Float32Array(4).fill(0), [1, 4]);

    const norm = () =>
      toArr(candle.linearForward(fc, x)).values.reduce((a, b) => a + b * b, 0);

    const before = norm();
    for (let i = 0; i < 50; i++) {
      const l = candle.mseLoss(candle.linearForward(fc, x), y);
      candle.adamwStep(opt, candle.backward(l));
    }
    expect(norm()).toBeLessThan(before);
  });

  test("adamw with no grads is a no-op, not a crash", () => {
    const vm = candle.varmapNew();
    candle.linearNew(vm, "fc", 2, 2, true);
    const opt = adamw(vm);

    // loss built from a var NOT in this varmap
    const w = candle.varFromF32(new Float32Array([2]), [1]);
    const loss = candle.tensorMul(w, w);
    expect(() => candle.adamwStep(opt, candle.backward(loss))).not.toThrow();
  });
});