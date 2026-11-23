// ESM entry point. The implementation is compiled to CommonJS in dist/; this
// wrapper hands ESM users the composed `candle` default and real named exports
// instead of Node's CJS namespace object (which lacks `nn` / `optim`).
import mod from "./dist/index.js";
import { Tensor, Var, GradStore } from "./dist/tensor.js";
import { VarMap } from "./dist/varmap.js";
import { expr } from "./dist/expr.js";

const candle = mod.default ?? mod;
const { F, nn, optim, hl } = candle;

export default candle;
export { Tensor, Var, GradStore, VarMap, expr, F, nn, optim, hl };
export * from "./dist/factory.js";
export * from "./dist/highlevel.js";
export * from "./dist/nn.js";
export * from "./dist/optim.js";
export * from "./dist/ops.js";
