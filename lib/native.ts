/**
 * Low-level bridge to the compiled `candle_js` native addon.
 *
 * Everything in this module is internal. The friendly, PyTorch-like API lives
 * in the other files under `lib/` and is re-exported from `lib/index.ts`.
 *
 * Resolution order:
 *   1. `index.node` at the package root (local `cargo build` output).
 *   2. `prebuilds/<platform>-<arch>/candle_js.node` (shipped in the npm tarball).
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare function require(id: string): any;

/**
 * The raw Neon exports. Values are native handles (opaque JS objects) unless
 * stated otherwise, so they are intentionally typed as `any` at this boundary.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type NativeAddon = Record<string, any>;

interface ProcLike {
  platform?: string;
  arch?: string;
}

function candidatePaths(): string[] {
  const paths = ["../index.node"];
  const proc = (globalThis as { process?: ProcLike }).process;
  if (proc?.platform && proc?.arch) {
    paths.push(`../prebuilds/${proc.platform}-${proc.arch}/candle_js.node`);
  }
  return paths;
}

function loadAddon(): NativeAddon {
  const errors: string[] = [];
  for (const path of candidatePaths()) {
    try {
      return require(path);
    } catch (err) {
      errors.push(`  ${path}: ${(err as Error).message}`);
    }
  }
  throw new Error(
    "candle_js: failed to load the native addon (index.node).\n" +
      "Build it from source with `bun run build`.\n" +
      "Tried:\n" +
      errors.join("\n")
  );
}

/** The loaded native addon. @internal */
export const native: NativeAddon = loadAddon();
