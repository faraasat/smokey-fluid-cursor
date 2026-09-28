import { defineConfig } from "tsup";
import { glslMinifyPlugin } from "./glsl-minify";

const shared = {
  entry: ["src/index.ts"],
  clean: false,
  target: "es2019",

  // `splitting` is incompatible with the iife build and buys nothing for a
  // single entry point.
  splitting: false,
  shims: false,

  // Shader source is ~40% of this bundle and lives in template literals, which
  // JS minifiers leave untouched. Strip GLSL comments/indentation at build time.
  esbuildPlugins: [glslMinifyPlugin()],
} as const;

export default defineConfig([
  {
    ...shared,
    format: ["cjs", "esm"],
    dts: true,
    clean: true,
    // Minified: this is what actually ships to a visitor's browser, and it
    // keeps the install footprint small. Build from source if you need to
    // step through it.
    minify: true,

    // Sourcemaps are deliberately not published. They were ~65% of the install
    // footprint, and this build is already readable. Build from source if you
    // need to step through it.
    sourcemap: false,
  },
  {
    ...shared,
    // The CDN build is loaded straight by the browser with no bundler in
    // front of it, so here minification is the whole point.
    format: ["iife"],
    globalName: "SmokeyFluid",
    dts: false,
    minify: true,
    sourcemap: false,
  },
]);
