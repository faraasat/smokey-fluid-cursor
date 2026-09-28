import { defineConfig, devices } from "@playwright/test";

/**
 * E2E against the built demo site.
 *
 * The demo is a static export, so the web server here just serves `out/`.
 * These tests are the layer unit tests cannot cover: real layout, real CSS,
 * real keyboard behaviour.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? "github" : "list",

  use: {
    baseURL: "http://127.0.0.1:4318",
    trace: "on-first-retry",
  },

  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],

  webServer: {
    // `npx serve` is intentionally not used: it is an extra dependency for
    // something `python3 -m http.server` already does in CI and locally.
    command: "npx --yes http-server example/out -p 4318 -s --silent",
    url: "http://127.0.0.1:4318",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
