import candle, { expr } from "../lib/index";

const a = candle.tensor([
  [1, 2],
  [3, 4],
]);
const b = candle.tensor([
  [5, 6],
  [7, 8],
]);
const c = candle.ones([2, 2]);

console.log("a @ b + c:", expr`${a} @ ${b} + ${c}`.tolist());
console.log("(a + b) * c:", expr`(${a} + ${b}) * ${c}`.tolist());
console.log("2 * a ^ 2:", expr`2 * ${a} ^ 2`.tolist());

const w = candle.variable([3], [1]);
console.log("autograd through expr:", expr`${w} * ${w}`.backward().get(w).item());
