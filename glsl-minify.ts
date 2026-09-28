import type { Plugin } from "esbuild";

/**
 * Strips comments and redundant whitespace from GLSL written as template
 * literals.
 *
 * JS minifiers never touch string contents, so every comment and every level
 * of indentation inside a shader is shipped verbatim to the browser. These
 * shaders are ~40% of this bundle, so that is worth reclaiming.
 *
 * Deliberately conservative: it only rewrites literals that look like GLSL
 * (they declare a `precision`, or a shader-only qualifier), never interpolates
 * across `${}`, and preserves `#version` / `#define` lines intact.
 */
const minifyGlsl = (src: string): string =>
  src
    // block and line comments
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\/\/[^\n]*/g, "")
    // collapse indentation, keeping preprocessor directives on their own lines
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .join("\n")
    // Collapse runs of spaces inside a line, but leave token separation alone.
    //
    // Deliberately NOT stripping spaces around operators: `a - -b` would
    // become `a--b`, which GLSL parses as the decrement operator. The
    // remaining win is not worth miscompiling a shader for.
    .replace(/[ \t]{2,}/g, " ")
    .trim();

const GLSL_HINT = /\b(precision\s+(?:lowp|mediump|highp)|gl_FragColor|gl_Position|varying\s|uniform\s+sampler2D)\b/;

export const glslMinifyPlugin = (): Plugin => ({
  name: "glsl-minify",
  setup(build) {
    build.onLoad({ filter: /\.tsx?$/ }, async (args) => {
      const fs = await import("node:fs/promises");
      const source = await fs.readFile(args.path, "utf8");
      if (!GLSL_HINT.test(source)) return null;

      let saved = 0;
      const out = source.replace(/`([^`\\]*)`/g, (whole, body: string) => {
        // Skip anything with interpolation or that does not look like GLSL.
        if (body.includes("${") || !GLSL_HINT.test(body)) return whole;
        const min = minifyGlsl(body);
        saved += body.length - min.length;
        return "`" + min + "`";
      });

      if (saved > 0) {
        console.log(`  glsl-minify: ${args.path.split("/").pop()} -${(saved / 1024).toFixed(1)} kB`);
      }
      return { contents: out, loader: args.path.endsWith(".tsx") ? "tsx" : "ts" };
    });
  },
});
