import { expect, test, describe } from "bun:test";
import candle, { expr } from "../lib/index";

describe("expr: operator-style DSL", () => {
  test("matmul and add", () => {
    const a = candle.tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = candle.tensor([
      [5, 6],
      [7, 8],
    ]);
    const c = candle.ones([2, 2]);
    expect(expr`${a} @ ${b} + ${c}`.tolist()).toEqual([
      [20, 23],
      [44, 51],
    ]);
  });

  test("precedence and parentheses", () => {
    const a = candle.tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = candle.tensor([
      [5, 6],
      [7, 8],
    ]);
    const c = candle.tensor([
      [2, 1],
      [1, 2],
    ]);
    // a + (b * c): elementwise b*c = [[10,6],[7,16]], then + a
    expect(expr`${a} + ${b} * ${c}`.tolist()).toEqual([
      [11, 8],
      [10, 20],
    ]);
    // (a + b) * c
    expect(expr`(${a} + ${b}) * ${c}`.tolist()).toEqual([
      [12, 8],
      [10, 24],
    ]);
  });

  test("scalars on the left and powers", () => {
    const x = candle.tensor([1, 2, 3]);
    expect(expr`2 * ${x}`.toArray()).toEqual([2, 4, 6]);
    expect(expr`${x} ^ 2`.toArray()).toEqual([1, 4, 9]);
    expect(expr`${x} ** 2`.toArray()).toEqual([1, 4, 9]);
    expect(expr`10 - ${x}`.toArray()).toEqual([9, 8, 7]);
    expect(expr`6 / ${x}`.toArray()).toEqual([6, 3, 2]);
  });

  test("autograd flows through expr", () => {
    const w = candle.variable([3], [1]);
    const g = expr`${w} * ${w}`.backward().get(w);
    expect(g.item()).toBeCloseTo(6, 4);
  });

  test("malformed expressions throw", () => {
    const a = candle.tensor([1]);
    expect(() => expr`${a} + `).toThrow();
    expect(() => expr`(${a}`).toThrow();
  });
});
