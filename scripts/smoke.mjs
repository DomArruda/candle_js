// Smoke test for the packaged artifact.
//
// Loads the ESM entry (index.mjs) and the CommonJS build (dist/index.js), then
// runs real ops through the native addon. Run with:
//   bun scripts/smoke.mjs
//   node scripts/smoke.mjs
import { createRequire } from "node:module";
import candle, {
  F,
  Linear,
  Tensor,
  expr,
  fit,
  hl,
  nn,
  tensor as namedTensor,
  trainTestSplit,
} from "../index.mjs";

const require = createRequire(import.meta.url);
const cjs = require("../dist/index.js").default;

const a = candle.tensor([
  [1, 2],
  [3, 4],
]);
const b = candle.ones([2, 2]);

console.log("shape:", a.shape);
console.log("matmul:", a.matmul(b).tolist());

const w = candle.variable([3], [1]);
console.log("grad:", w.mul(w).backward().get(w).item());

console.log("esm nn.functional:", typeof candle.nn.functional.mseLoss);
console.log("cjs nn.functional:", typeof cjs.nn.functional.mseLoss);

if (!(a instanceof Tensor)) throw new Error("ESM Tensor export is wrong");
if (namedTensor([1]).item() !== 1) throw new Error("ESM named factory is wrong");
if (typeof Linear !== "function") throw new Error("ESM named layer is wrong");
if (expr`${a} + 1`.shape.join() !== "2,2") throw new Error("ESM expr is wrong");
if (!(nn.Linear && F.mseLoss)) throw new Error("ESM named exports are wrong");
if (typeof fit !== "function" || typeof trainTestSplit !== "function") {
  throw new Error("ESM high-level exports are wrong");
}
if (candle.fit !== fit || hl.fit !== fit || candle.hl !== hl) {
  throw new Error("ESM high-level namespaces are wrong");
}
if (F.mseLoss(candle.tensor([1]), candle.tensor([1])).item() !== 0) {
  throw new Error("ESM functional ops are wrong");
}
if (typeof cjs.nn.functional.mseLoss !== "function") {
  throw new Error("CJS default entry is wrong");
}
if (typeof cjs.fit !== "function" || typeof cjs.trainTestSplit !== "function") {
  throw new Error("CJS high-level entry is wrong");
}

console.log("smoke ok");
