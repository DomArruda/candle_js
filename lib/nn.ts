import { native } from "./native";
import { Tensor, Var, unwrap, type ShapeLike, toShape } from "./tensor";
import { VarMap } from "./varmap";
import * as F from "./ops";

/** @internal Join a module path with a child name. */
function joinPrefix(prefix: string, name: string): string {
  return prefix === "" ? name : `${prefix}.${name}`;
}

/**
 * Base class for all neural-network layers.
 *
 * A module owns a {@link VarMap} and a dotted name for every parameter, which
 * makes {@link Module.parameters}, {@link Module.save} and
 * {@link Module.load} work across arbitrarily nested models — the equivalent
 * of a PyTorch `nn.Module` and its `state_dict`.
 *
 * Subclass it to build custom architectures. Create layers as fields with
 * `this.addModule(...)` so they are registered, then implement `_forward`:
 *
 * @example
 * ```ts
 * import candle, { Tensor } from "candle_js";
 *
 * class MLP extends candle.nn.Module {
 *   fc1 = this.addModule(new candle.nn.Linear(2, 8));
 *   fc2 = this.addModule(new candle.nn.Linear(8, 1));
 *
 *   _forward(x: Tensor): Tensor {
 *     return this.fc2.forward(this.fc1.forward(x).tanh());
 *   }
 * }
 *
 * const model = new MLP();
 * const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.05 });
 * ```
 */
export abstract class Module {
  private _vm?: VarMap;
  private _built = false;
  private _params: Var[] = [];
  private _children: Module[] = [];

  /** Whether the module is in training mode. Affects {@link Dropout}. */
  training = true;

  /** @internal Build this module and its descendants against a shared map. */
  _build(vm: VarMap, prefix: string): void {
    if (this._built) return;
    this._built = true;
    this._vm = vm;
    this.build(vm, prefix);
    this._children.forEach((child, i) =>
      child._build(vm, joinPrefix(prefix, String(i)))
    );
  }

  /** Subclass hook: create native layers and register parameters. */
  protected build(_vm: VarMap, _prefix: string): void {}

  /** Register a parameter created inside {@link Module.build}. */
  protected registerParameter(v: Var): void {
    this._params.push(v);
  }

  /** Register a child module; returns it so it can be assigned to a field. */
  protected addModule<M extends Module>(m: M): M {
    this._children.push(m);
    return m;
  }

  /** @internal Create a map and names on first use. */
  protected ensureBuilt(): void {
    if (!this._built) this._build(new VarMap(), "");
  }

  /** The variable map that owns this module's parameters. */
  get varmap(): VarMap {
    this.ensureBuilt();
    return this._vm!;
  }

  /** Every parameter in this module and its descendants, in registration order. */
  parameters(): Var[] {
    this.ensureBuilt();
    return [...this._params, ...this._children.flatMap((c) => c.parameters())];
  }

  /**
   * Run the forward pass. Ensures the module is built, then calls
   * {@link Module._forward} with the same arguments.
   */
  forward(...args: any[]): Tensor {
    this.ensureBuilt();
    return this._forward(...args);
  }

  /** The forward computation implemented by subclasses. */
  protected abstract _forward(...args: any[]): Tensor;

  /** Set training mode recursively. */
  train(mode = true): this {
    this.training = mode;
    this._children.forEach((c) => c.train(mode));
    return this;
  }

  /** Set evaluation mode recursively (turns off {@link Dropout}). */
  eval(): this {
    return this.train(false);
  }

  /** Save this module's parameters as safetensors. */
  save(path: string): void {
    this.varmap.save(path);
  }

  /** Load this module's parameters from a safetensors file. */
  load(path: string): void {
    this.varmap.load(path);
  }
}

/** Options for {@link Linear}. */
export interface LinearOptions {
  /** Whether to include a bias term (default `true`). */
  bias?: boolean;
}

/** A fully-connected layer: `y = x @ weight.T + bias`. */
export class Linear extends Module {
  private _layer: unknown;
  private _inFeatures: number;
  private _outFeatures: number;
  private _bias: boolean;

  /**
   * @param inFeatures - Size of the input's last dimension.
   * @param outFeatures - Size of the output's last dimension.
   * @param options - Optional settings, e.g. `{ bias: false }`.
   * @example
   * ```ts
   * new candle.nn.Linear(128, 512);              // with bias
   * new candle.nn.Linear(128, 512, { bias: false });
   * ```
   */
  constructor(
    inFeatures: number,
    outFeatures: number,
    options: LinearOptions = {}
  ) {
    super();
    this._inFeatures = inFeatures;
    this._outFeatures = outFeatures;
    this._bias = options.bias ?? true;
  }

  protected build(vm: VarMap, prefix: string): void {
    const name = prefix || "linear";
    this._layer = native.linearNew(
      vm._handle,
      name,
      this._inFeatures,
      this._outFeatures,
      this._bias
    );
    this.registerParameter(new Var(native.varmapGet(vm._handle, `${name}.weight`)));
    if (this._bias) {
      this.registerParameter(new Var(native.varmapGet(vm._handle, `${name}.bias`)));
    }
  }

  protected _forward(input: Tensor): Tensor {
    return new Tensor(native.linearForward(this._layer, unwrap(input)));
  }
}

/** A lookup table mapping integer ids to dense vectors. */
export class Embedding extends Module {
  private _layer: unknown;

  /**
   * @param vocabSize - Number of distinct ids.
   * @param embeddingDim - Size of each embedding vector.
   */
  constructor(private _vocabSize: number, private _embeddingDim: number) {
    super();
  }

  protected build(vm: VarMap, prefix: string): void {
    const name = prefix || "embedding";
    this._layer = native.embeddingNew(
      vm._handle,
      name,
      this._vocabSize,
      this._embeddingDim
    );
    this.registerParameter(new Var(native.varmapGet(vm._handle, `${name}.weight`)));
  }

  /**
   * @param ids - Token ids.
   * @param shape - Shape of `ids` (defaults to `[ids.length]`); the output is
   *   `[...shape, embeddingDim]`.
   */
  protected _forward(ids: Uint32Array | number[], shape?: ShapeLike): Tensor {
    const arr = ids instanceof Uint32Array ? ids : new Uint32Array(ids);
    const s = shape === undefined ? [arr.length] : toShape(shape);
    return new Tensor(native.embeddingForward(this._layer, arr, s));
  }
}

/** Layer normalization over the last dimension. */
export class LayerNorm extends Module {
  private _layer: unknown;

  /**
   * @param normalizedShape - Size of the last dimension.
   * @param options.eps - Numerical stability epsilon (default `1e-5`).
   * @param options.affine - Learn a weight (and bias) (default `true`).
   */
  constructor(
    private _normalizedShape: number,
    private _options: { eps?: number; affine?: boolean } = {}
  ) {
    super();
  }

  protected build(vm: VarMap, prefix: string): void {
    const name = prefix || "norm";
    const eps = this._options.eps ?? 1e-5;
    const affine = this._options.affine ?? true;
    this._layer = native.layerNormNew(
      vm._handle,
      name,
      this._normalizedShape,
      eps,
      affine
    );
    this.registerParameter(new Var(native.varmapGet(vm._handle, `${name}.weight`)));
    if (affine) {
      this.registerParameter(new Var(native.varmapGet(vm._handle, `${name}.bias`)));
    }
  }

  protected _forward(input: Tensor): Tensor {
    return new Tensor(native.layerNormForward(this._layer, unwrap(input)));
  }
}

/** Root-mean-square normalization over the last dimension (no mean removal). */
export class RMSNorm extends Module {
  private _layer: unknown;

  /**
   * @param normalizedShape - Size of the last dimension.
   * @param eps - Numerical stability epsilon (default `1e-5`).
   */
  constructor(private _normalizedShape: number, private _eps = 1e-5) {
    super();
  }

  protected build(vm: VarMap, prefix: string): void {
    const name = prefix || "norm";
    this._layer = native.rmsNormNew(
      vm._handle,
      name,
      this._normalizedShape,
      this._eps
    );
    this.registerParameter(new Var(native.varmapGet(vm._handle, `${name}.weight`)));
  }

  protected _forward(input: Tensor): Tensor {
    return new Tensor(native.rmsNormForward(this._layer, unwrap(input)));
  }
}

/** Inverted dropout. A no-op in {@link Module.eval} mode. */
export class Dropout extends Module {
  private _layer: unknown;

  /** @param p - Probability of dropping an element (default `0.5`). */
  constructor(private _p = 0.5) {
    super();
  }

  protected build(): void {
    this._layer = native.dropoutNew(this._p);
  }

  protected _forward(input: Tensor): Tensor {
    return new Tensor(
      native.dropoutForward(this._layer, unwrap(input), this.training)
    );
  }
}

/** Options for {@link LSTM}. */
export interface LSTMOptions {
  /** Whether to include input/recurrent bias terms (default `true`). */
  bias?: boolean;
}

/**
 * A single-layer Long Short-Term Memory layer.
 *
 * Input shape `[batch, seq, inFeatures]`, output shape
 * `[batch, seq, hiddenSize]` — the hidden state at every timestep. Backed by
 * candle's `nn::rnn::LSTM` and fully differentiable.
 *
 * @example
 * ```ts
 * const lstm = new candle.nn.LSTM(1, 16);
 * const out = lstm.forward(seq); // [B, T, 16]
 * ```
 */
export class LSTM extends Module {
  private _layer: unknown;
  private _inFeatures: number;
  private _hiddenSize: number;
  private _bias: boolean;

  /**
   * @param inFeatures - Size of each input vector.
   * @param hiddenSize - Size of the hidden state.
   * @param options - Optional `{ bias: false }`.
   */
  constructor(inFeatures: number, hiddenSize: number, options: LSTMOptions = {}) {
    super();
    this._inFeatures = inFeatures;
    this._hiddenSize = hiddenSize;
    this._bias = options.bias ?? true;
  }

  protected build(vm: VarMap, prefix: string): void {
    const name = prefix || "lstm";
    this._layer = native.lstmNew(
      vm._handle,
      name,
      this._inFeatures,
      this._hiddenSize,
      this._bias
    );
    this.registerParameter(
      new Var(native.varmapGet(vm._handle, `${name}.weight_ih_l0`))
    );
    this.registerParameter(
      new Var(native.varmapGet(vm._handle, `${name}.weight_hh_l0`))
    );
    if (this._bias) {
      this.registerParameter(
        new Var(native.varmapGet(vm._handle, `${name}.bias_ih_l0`))
      );
      this.registerParameter(
        new Var(native.varmapGet(vm._handle, `${name}.bias_hh_l0`))
      );
    }
  }

  protected _forward(input: Tensor): Tensor {
    return new Tensor(native.lstmSeq(this._layer, unwrap(input)));
  }
}

/** @internal Shared implementation for parameter-free activation modules. */
abstract class Activation extends Module {
  protected abstract apply(x: Tensor): Tensor;

  protected _forward(input: Tensor): Tensor {
    return this.apply(input);
  }
}

/** `ReLU(x) = max(x, 0)`. */
export class ReLU extends Activation {
  protected apply(x: Tensor): Tensor {
    return F.relu(x);
  }
}

/** Gaussian error linear unit (tanh approximation). */
export class GELU extends Activation {
  protected apply(x: Tensor): Tensor {
    return F.gelu(x);
  }
}

/** Sigmoid linear unit (SiLU / swish). */
export class SiLU extends Activation {
  protected apply(x: Tensor): Tensor {
    return F.silu(x);
  }
}

/** Logistic sigmoid. */
export class Sigmoid extends Activation {
  protected apply(x: Tensor): Tensor {
    return F.sigmoid(x);
  }
}

/** Hyperbolic tangent. */
export class Tanh extends Activation {
  protected apply(x: Tensor): Tensor {
    return F.tanh(x);
  }
}

/** Mish activation. */
export class Mish extends Activation {
  protected apply(x: Tensor): Tensor {
    return F.mish(x);
  }
}

/** Leaky ReLU with a configurable negative slope. */
export class LeakyReLU extends Activation {
  /** @param negativeSlope - Slope for negative inputs (default `0.01`). */
  constructor(private _negativeSlope = 0.01) {
    super();
  }

  protected apply(x: Tensor): Tensor {
    return F.leakyRelu(x, this._negativeSlope);
  }
}

/** Exponential linear unit. */
export class ELU extends Activation {
  /** @param alpha - Scale for negative inputs (default `1.0`). */
  constructor(private _alpha = 1.0) {
    super();
  }

  protected apply(x: Tensor): Tensor {
    return F.elu(x, this._alpha);
  }
}

/**
 * Wraps an arbitrary function as a module, for use inside `Sequential`.
 *
 * @example `new candle.nn.Lambda((x) => x.gelu())`
 */
export class Lambda extends Activation {
  constructor(private _fn: (x: Tensor) => Tensor) {
    super();
  }

  protected apply(x: Tensor): Tensor {
    return this._fn(x);
  }
}

/** Runs a list of modules in order. */
export class Sequential extends Module {
  private _layers: Module[];

  /**
   * @param layers - Modules applied left to right.
   * @example `new candle.nn.Sequential(new candle.nn.Linear(2, 8), new candle.nn.Tanh(), new candle.nn.Linear(8, 1))`
   */
  constructor(...layers: Module[]) {
    super();
    this._layers = layers.map((layer) => this.addModule(layer));
  }

  protected _forward(input: Tensor): Tensor {
    return this._layers.reduce((acc, layer) => layer.forward(acc), input);
  }
}
