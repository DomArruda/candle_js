// Smoke-test the wasm32 Node build produced by examples/browser/build.sh.
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { matmul, train_xor, version } = require("./pkg-node/candle_wasm.js");

const product = Array.from(
  matmul(new Float32Array([1, 2, 3, 4]), 2, 2, new Float32Array([5, 6, 7, 8]), 2, 2)
);
if (product.join() !== "19,22,43,50") {
  throw new Error(`matmul mismatch: ${product}`);
}

const xor = Array.from(train_xor(300)).map((v) => Math.round(v));
if (xor.join() !== "0,1,1,0") {
  throw new Error(`xor mismatch: ${xor}`);
}

console.log(version());
console.log("wasm matmul:", product);
console.log("wasm xor:", xor);
