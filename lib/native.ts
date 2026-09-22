/**
 * Low-level bridge to the compiled `candle_js` native addon.
 *
 * Everything in this module is internal. The friendly, PyTorch-like API lives in
 * the other files under `lib/` and is re-exported from `lib/index.ts`.
 *
 * The addon is loaded once, from `index.node` at the package root. Build it
 * with `cargo build --release && cp target/release/libcandle_js.so index.node`.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare function require(id: string): any;

/**
 * The raw Neon exports. Values are native handles (opaque JS objects) unless
 * stated otherwise, so they are intentionally typed as `any` at this boundary.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type NativeAddon = Record<string, any>;

let addon: NativeAddon;
try {
  addon = require("../index.node");
} catch (err) {
  throw new Error(
    "candle_js: failed to load the native addon (index.node).\n" +
      "Build it first:\n" +
      "  cargo build --release && cp target/release/libcandle_js.so index.node\n" +
      "Original error: " +
      (err as Error).message
  );
}

/** The loaded native addon. @internal */
export const native: NativeAddon = addon;
