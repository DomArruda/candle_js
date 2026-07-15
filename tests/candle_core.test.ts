import { expect, test, describe } from "bun:test";
const candle = require('./index.node')

const toArr = (t: any) => {
  const { shape, data } = candle.tensorToF32(t);
  return { shape, values: Array.from(new Float32Array(data)) };
};

describe("candle", () => {
  test("hello/add", () => {
    expect(candle.hello()).toBe("hello from candle");
    expect(candle.add(2, 3)).toBe(5);
  });

  test("round-trips", () => {
    const t = candle.tensorFromF32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    expect(candle.tensorShape(t)).toEqual([2, 3]);
    expect(toArr(t)).toEqual({ shape: [2, 3], values: [1, 2, 3, 4, 5, 6] });
  });

  test("add", () => {
    const a = candle.tensorFromF32(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const b = candle.tensorFromF32(new Float32Array([10, 20, 30, 40]), [2, 2]);
    expect(toArr(candle.tensorAdd(a, b)).values).toEqual([11, 22, 33, 44]);
  });

  test("matmul", () => {
    const a = candle.tensorFromF32(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const b = candle.tensorFromF32(new Float32Array([5, 6, 7, 8]), [2, 2]);
    expect(toArr(candle.tensorMatmul(a, b)).values).toEqual([19, 22, 43, 50]);
  });

  test("dot", () => {
    const a = candle.tensorFromF32(new Float32Array([1, 2, 3]), [3]);
    const b = candle.tensorFromF32(new Float32Array([4, 5, 6]), [3]);
    expect(candle.dot(a, b)).toBeCloseTo(32, 5);
  });

  test("throws on bad matmul", () => {
    const a = candle.tensorFromF32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    expect(() => candle.tensorMatmul(a, a)).toThrow();
  });
});