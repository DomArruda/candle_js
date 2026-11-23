// Run every example sequentially and report pass/fail.
//
//   bun run examples
import { spawnSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { join } from "node:path";

const dir = join(import.meta.dir, "..", "examples");
const files = readdirSync(dir)
  .filter((f) => f.endsWith(".ts"))
  .sort();

let failed = 0;
const started = performance.now();
for (const file of files) {
  const start = performance.now();
  const result = spawnSync(process.execPath, [join(dir, file)], {
    stdio: "inherit",
  });
  const ms = performance.now() - start;
  const ok = result.status === 0;
  if (!ok) failed++;
  console.log(`\n[${ok ? "ok" : "FAIL"}] ${file} (${ms.toFixed(0)} ms)`);
}

const total = ((performance.now() - started) / 1000).toFixed(2);
if (failed > 0) {
  console.error(`\n${failed}/${files.length} examples failed in ${total}s`);
  process.exit(1);
}
console.log(`\nall ${files.length} examples passed in ${total}s`);
