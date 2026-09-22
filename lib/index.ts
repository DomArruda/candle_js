/**
 * candle_js — a friendly, PyTorch-like neural-network library for JavaScript and
 * TypeScript, backed by [candle](https://github.com/huggingface/candle) in Rust.
 *
 * The package has two layers:
 *
 * - **This library** (`lib/`) wraps the raw native exports with tensors that
 *   have methods, a `nn.Module` system, optimizers and documented functional
 *   ops.
 * - **The native addon** (`index.node`) is the thin FFI layer in `src/lib.rs`.
 *
 * Import the default export as `candle`, or use the named exports.
 *
 * @example
 * ```ts
 * import candle from "candle_js";
 *
 * const a = candle.tensor([[1, 2], [3, 4]]);
 * const b = candle.ones([2, 2]);
 * a.matmul(b).tolist(); // [[3, 3], [7, 7]]
 * ```
 *
 * @example
 * ```ts
 * import candle, { Tensor } from "candle_js";
 *
 * class MLP extends candle.nn.Module {
 *   fc1 = this.addModule(new candle.nn.Linear(2, 8));
 *   fc2 = this.addModule(new candle.nn.Linear(8, 1));
 *   _forward(x: Tensor) {
 *     return this.fc2.forward(this.fc1.forward(x).tanh());
 *   }
 * }
 *
 * const model = new MLP();
 * const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.05 });
 * const loss = candle.nn.functional.mseLoss(model.forward(X), Y);
 * opt.step(loss.backward());
 * ```
 */

import * as factory from "./factory";
import * as F from "./ops";
import * as nn from "./nn";
import * as optim from "./optim";
import { expr } from "./expr";
import { GradStore, Tensor, Var } from "./tensor";
import { VarMap } from "./varmap";

export { GradStore, Tensor, Var } from "./tensor";
export type { DimsLike, ShapeLike, TensorOrScalar } from "./tensor";
export { VarMap } from "./varmap";
export { expr } from "./expr";
export type { ExprOperand } from "./expr";
export * from "./factory";
export * from "./nn";
export * from "./optim";
export * from "./ops";

/** Functional ops, also available as `candle.nn.functional`. */
export { F };

/**
 * The `candle` namespace: creation functions at the top level, plus `nn`,
 * `optim` and `functional` sub-namespaces.
 */
const candle = {
  ...factory,
  F,
  expr,
  nn: { ...nn, functional: F },
  optim,
  Tensor,
  Var,
  GradStore,
  VarMap,
};

export default candle;
