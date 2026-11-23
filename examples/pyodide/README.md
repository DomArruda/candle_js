# Pyodide example

Runs CPython in [Pyodide](https://pyodide.org) on Node and calls candle_js
through the JS bridge. This exercises the real native addon, not a WASM build.

```bash
bun install
bun run build:js          # produces dist/ for the demo to import
bun run pyodide           # or: node examples/pyodide/demo.mjs
```

Python side:

```python
from pyodide.ffi import to_js
import js

candle = js.require(candle_path)
a = candle.tensor(to_js([[1.0, 2.0], [3.0, 4.0]]))
a.matmul(candle.ones(to_js([2, 2]))).tolist()
```

`to_js` matters: Neon expects real JS arrays, not Python list proxies.

A browser Pyodide page needs candle compiled to `wasm32-unknown-unknown`
(candle supports WASM on CPU). That build is not part of this package yet.
