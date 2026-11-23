import candle from "../lib/index";

// d/dw (w^2) at w = 3 is 2w = 6.
const w = candle.variable([3], [1]);
const grads = w.mul(w).backward();
console.log("d/dw w^2 at w=3:", grads.get(w).item());

// d/dp MSE(x @ p.T, 1) for a 1x2 p.
const x = candle.tensor([
  [1, 2],
  [3, 4],
]);
const p = candle.variable([0, 0], [1, 2]);
const target = candle.ones([2, 1]);
const loss = candle.F.mseLoss(x.matmul(p.t()), target);
console.log("loss:", loss.item().toFixed(4));
console.log("dL/dp:", loss.backward().get(p).tolist());
