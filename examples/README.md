# Examples

Runnable, self-contained scripts. From the repo root:

```bash
bun examples/01-tensors.ts
bun run examples        # run everything
```

| File | Shows |
| --- | --- |
| `01-tensors.ts` | creation, shape, matmul, reductions, readback |
| `02-autograd.ts` | gradients with `backward()` and `GradStore` |
| `03-regression.ts` | manual training loop fitting `sin` |
| `04-classification.ts` | two-blob classification |
| `05-highlevel.ts` | `model.fit`, validation, `evaluate`, `predict` |
| `06-xor.ts` | XOR with the high-level API |
| `07-lstm.ts` | LSTM running-sum regression |
| `08-transformer.ts` | causal self-attention block overfitting a sequence |
| `09-autoencoder.ts` | linear autoencoder on low-rank data |
| `10-text.ts` | embedding-bag text classification |
| `11-rl-bandit.ts` | REINFORCE on a contextual bandit |
| `12-rl-gridworld.ts` | REINFORCE on a tabular gridworld |
| `13-expr.ts` | the `expr` operator DSL |
| `14-save-load.ts` | safetensors round-trip |
| `15-benchmark.ts` | matmul and elementwise timings |
| `16-raw-native.ts` | the low-level native addon |

`pyodide/` demonstrates calling the library from Python through Pyodide.
`browser/` builds the WASM module and a JS + Pyodide browser demo.
`bun run examples` runs every `.ts` file above.
