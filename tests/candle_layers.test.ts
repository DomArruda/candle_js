import { expect, test, describe } from "bun:test";
const candle = require("../index.node");

const toArr = (t: any) => {
  const { shape, data } = candle.tensorToF32(t);
  return { shape, values: Array.from(new Float32Array(data)) };
};

describe("layers", () => {
  test("linear output shape", () => {
    const vm = candle.varmapNew();
    const l = candle.linearNew(vm, "fc", 3, 5, true);
    const x = candle.tensorFromF32(new Float32Array(6).fill(1), [2, 3]);
    expect(toArr(candle.linearForward(l, x)).shape).toEqual([2, 5]);
  });

  test("linear_no_bias also works", () => {
    const vm = candle.varmapNew();
    const l = candle.linearNew(vm, "fc", 3, 2, false);
    const x = candle.tensorFromF32(new Float32Array([1, 2, 3]), [1, 3]);
    expect(toArr(candle.linearForward(l, x)).shape).toEqual([1, 2]);
  });

  test("embedding looks up rows", () => {
    const vm = candle.varmapNew();
    const e = candle.embeddingNew(vm, "emb", 10, 4);
    const y = candle.embeddingForward(e, new Uint32Array([1, 5, 5]), [3]);
    const { shape, values } = toArr(y);
    expect(shape).toEqual([3, 4]);
    expect(values.slice(4, 8)).toEqual(values.slice(8, 12)); // same id -> same vec
  });

  test("embedding is trainable", () => {
    const vm = candle.varmapNew();
    const e = candle.embeddingNew(vm, "emb", 4, 2);
    const ids = new Uint32Array([0, 1]);

    const before = toArr(candle.embeddingForward(e, ids, [2])).values;
    const loss = candle.tensorSumAll(
      candle.tensorMul(
        candle.embeddingForward(e, ids, [2]),
        candle.embeddingForward(e, ids, [2])
      )
    );
    const n = candle.varmapSgdStep(vm, candle.backward(loss), 0.1);
    expect(n).toBe(1);

    const after = toArr(candle.embeddingForward(e, ids, [2])).values;
    expect(after).not.toEqual(before);
  });

  test("XOR with linear layers", () => {
    const vm = candle.varmapNew();
    const fc1 = candle.linearNew(vm, "fc1", 2, 8, true);
    const fc2 = candle.linearNew(vm, "fc2", 8, 1, true);

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
      candle.varmapSgdStep(vm, candle.backward(l), 0.1);
    }

    expect(loss).toBeLessThan(0.01);
    expect(toArr(forward(X)).values.map((v) => (v > 0.5 ? 1 : 0)))
      .toEqual([0, 1, 1, 0]);
  }, 30000);

  test("varmapSgdStep updates all 4 params", () => {
    const vm = candle.varmapNew();
    const fc1 = candle.linearNew(vm, "fc1", 2, 3, true);
    const fc2 = candle.linearNew(vm, "fc2", 3, 1, true);

    const x = candle.tensorFromF32(new Float32Array([1, 1]), [1, 2]);
    const y = candle.tensorFromF32(new Float32Array([1]), [1, 1]);
    const out = candle.linearForward(fc2, candle.tensorTanh(candle.linearForward(fc1, x)));

    const n = candle.varmapSgdStep(vm, candle.backward(candle.mseLoss(out, y)), 0.1);
    expect(n).toBe(4); // w1, b1, w2, b2
  });

  test("save/load round-trips weights", () => {
    const path = "/tmp/candle_test.safetensors";

    const vm = candle.varmapNew();
    const l = candle.linearNew(vm, "fc", 2, 3, true);
    const x = candle.tensorFromF32(new Float32Array([1, 2]), [1, 2]);
    const expected = toArr(candle.linearForward(l, x)).values;
    candle.varmapSave(vm, path);

    const vm2 = candle.varmapNew();
    const l2 = candle.linearNew(vm2, "fc", 2, 3, true); // must exist before load
    candle.varmapLoad(vm2, path);

    expect(toArr(candle.linearForward(l2, x)).values).toEqual(expected);
  });
});