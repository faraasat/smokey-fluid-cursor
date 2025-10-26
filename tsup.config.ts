import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["cjs", "esm", "iife"],
  dts: true,
  sourcemap: false,
  clean: true,
  splitting: true,
  target: "es2017",
  external: [],
  minify: true,
  shims: false,
  minifySyntax: true,
  minifyIdentifiers: true,
  minifyWhitespace: true,
  treeshake: true,
  globalName: "SmokyFluid",
});
