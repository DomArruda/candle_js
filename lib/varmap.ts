import { native } from "./native";
import { Var } from "./tensor";

/**
 * A name → {@link Var} table that owns a model's parameters and knows how to
 * persist them.
 *
 * Most users never build one by hand: {@link Module} creates a `VarMap`
 * automatically and names parameters after the module tree. Use one directly
 * when you want explicit control over names or to load a safetensors file.
 *
 * @example
 * ```ts
 * const vm = new candle.VarMap();
 * const fc = new candle.nn.Linear(2, 3);
 * // ... train ...
 * vm.save("model.safetensors");
 * ```
 */
export class VarMap {
  /** @internal Native handle. */
  readonly _handle: unknown;

  /** Create an empty variable map. */
  constructor() {
    this._handle = native.varmapNew();
  }

  /** Number of variables currently in the map. */
  get size(): number {
    return native.varmapNumVars(this._handle);
  }

  /**
   * Look up a variable by its full dotted name, e.g. `"0.weight"`.
   * @throws If no variable has that name.
   */
  get(name: string): Var {
    return new Var(native.varmapGet(this._handle, name));
  }

  /** Write every variable to a safetensors file. */
  save(path: string): void {
    native.varmapSave(this._handle, path);
  }

  /** Load values from a safetensors file into matching variables. */
  load(path: string): void {
    native.varmapLoad(this._handle, path);
  }
}
