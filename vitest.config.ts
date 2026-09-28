import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,

    // Playwright specs live in e2e/ and must not be collected by vitest:
    // they call Playwright's test() and fail with "did not expect test() to
    // be called here". They run via `npm run test:e2e`.
    include: ["test/**/*.{test,spec}.{ts,tsx}"],
    exclude: ["e2e/**", "node_modules/**", "dist/**", "example/**"],

    environment: "jsdom",
    coverage: { provider: "v8", reporter: ["text", "lcov"], include: ["src/**"] },
  },
});
