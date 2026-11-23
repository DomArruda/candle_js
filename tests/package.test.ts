import { existsSync } from "node:fs";
import { expect, test, describe } from "bun:test";
import pkg from "../package.json";

describe("package: npm and GitHub metadata", () => {
  test("entry points and shipped files are declared", () => {
    expect(pkg.exports["."].types).toBe("./dist/index.d.ts");
    expect(pkg.exports["."].require).toBe("./dist/index.js");
    expect(pkg.exports["."].import).toBe("./index.mjs");

    for (const file of ["dist", "index.mjs", "prebuilds", "LICENSE", "README.md"]) {
      expect(pkg.files).toContain(file);
    }
    expect(existsSync(new URL("../index.mjs", import.meta.url))).toBe(true);
    expect(existsSync(new URL("../LICENSE", import.meta.url))).toBe(true);
  });

  test("repository, license and scripts are set", () => {
    expect(pkg.repository.url).toContain("github.com/DomArruda/candle_js");
    expect(pkg.homepage).toContain("github.com/DomArruda/candle_js");
    expect(pkg.bugs).toContain("github.com/DomArruda/candle_js/issues");
    expect(pkg.license).toBe("MIT");
    expect(pkg.publishConfig.access).toBe("public");
    expect(pkg.scripts.smoke).toContain("smoke.mjs");
    expect(pkg.scripts.examples).toContain("run-examples");
    expect(pkg.scripts["bench:tfjs"]).toContain("bench-vs-tfjs");
    expect(pkg.scripts.pyodide).toContain("demo.mjs");
    expect(pkg.scripts.wasm).toContain("build.sh");
  });
});
