// Check the Pyodide -> wasm32 bridge used by examples/browser/index.html.
//
// Run after `bash examples/browser/build.sh` and `bun install`:
//   node examples/browser/pyodide-smoke.mjs
import { createRequire } from "node:module";
import { loadPyodide } from "pyodide";

const require = createRequire(import.meta.url);
const wasm = require("./pkg-node/candle_wasm.js");

const py = await loadPyodide();
py.globals.set("candle_wasm", wasm);
const result = await py.runPythonAsync(`
from pyodide.ffi import to_js
import js

a = js.Float32Array.new(to_js([1.0, 2.0, 3.0, 4.0]))
b = js.Float32Array.new(to_js([5.0, 6.0, 7.0, 8.0]))
product = [float(v) for v in candle_wasm.matmul(a, 2, 2, b, 2, 2)]
predictions = [round(float(v), 2) for v in candle_wasm.train_xor(300)]

f"matmul={product}\\nxor={predictions}"
`);

if (!result.includes("matmul=[19.0, 22.0, 43.0, 50.0]")) {
  throw new Error(`wasm matmul mismatch:\n${result}`);
}
if (!result.includes("xor=[0.0, 1.0, 1.0, 0.0]")) {
  throw new Error(`wasm xor mismatch:\n${result}`);
}
console.log(result);
console.log("wasm pyodide smoke ok");
