import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["cjs", "esm", "iife"],
  dts: true,
  clean: true,
  target: "es2019",
  globalName: "SmokeyFluid",

  // `splitting` is incompatible with the iife build and buys nothing for a
  // single entry point.
  splitting: false,

  minify: false,
  sourcemap: true,
  shims: false,
});
