// Run Python inside Pyodide (on Node) and call candle_js from Python.
//
//   bun run build:js     # once, to produce dist/
//   bun run pyodide
//
// Pyodide exposes Node's require to Python, so this uses the real native
// addon. A browser build needs the WASM target; see the root README.
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "pyodide";

const require = createRequire(import.meta.url);
globalThis.require = require;
const dist = fileURLToPath(new URL("../../dist/index.js", import.meta.url));

const py = await loadPyodide();
py.globals.set("candle_path", dist);

const code = `
from pyodide.ffi import to_js
import js

candle = js.require(candle_path)
opts = lambda d: to_js(d, dict_converter=js.Object.fromEntries)

a = candle.tensor(to_js([[1.0, 2.0], [3.0, 4.0]]))
b = candle.ones(to_js([2, 2]))
print("a @ b:", js.JSON.stringify(a.matmul(b).tolist()))

w = candle.variable(to_js([3.0]), to_js([1]))
print("d/dw w^2 at w=3:", w.mul(w).backward().get(w).item())

X = candle.tensor(to_js([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
Y = candle.tensor(to_js([[0.0], [1.0], [1.0], [0.0]]))
model = candle.nn.Sequential.new(
    candle.nn.Linear.new(2, 8),
    candle.nn.Tanh.new(),
    candle.nn.Linear.new(8, 1),
)
opt = candle.optim.AdamW.new(model.parameters(), opts({"lr": 0.05}))
for step in range(300):
    loss = candle.nn.functional.mseLoss(model.forward(X), Y)
    opt.step(loss.backward())

print("xor after 300 steps:", [round(v) for v in model.forward(X).toArray()])
"pyodide ok"
`;

const result = await py.runPythonAsync(code);
console.log("python returned:", result);
