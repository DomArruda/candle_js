// Copies the freshly built cdylib into the layout npm expects.
//
//   target/release/libcandle_js.{so,dylib,dll}
//     -> prebuilds/<platform>-<arch>/candle_js.node
//     -> index.node            (so local dev + tests keep working)
//
// Run after `cargo build --release`; `bun run build` does both.
import { copyFileSync, mkdirSync } from "node:fs";
import { arch, platform } from "node:process";

const ext = platform === "win32" ? "dll" : platform === "darwin" ? "dylib" : "so";
const src = `target/release/libcandle_js.${ext}`;
const dir = `prebuilds/${platform}-${arch}`;

mkdirSync(dir, { recursive: true });
copyFileSync(src, `${dir}/candle_js.node`);
copyFileSync(src, "index.node");

console.log(`native addon -> ${dir}/candle_js.node (+ index.node)`);
