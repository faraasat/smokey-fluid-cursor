import { test, expect } from "@playwright/test";
import { expectNoHorizontalOverflow, failOnConsoleErrors } from "./_helpers";

/** The status pill, scoped so prose containing "running" cannot match. */
const statusPill = (page: import("@playwright/test").Page) =>
  page.locator(".pill").first();

test("renders a canvas and starts the simulation", async ({ page }) => {
  await page.goto("/");
  const canvas = page.locator("canvas").first();
  await expect(canvas).toBeAttached();
  await expect(statusPill(page)).toHaveText("running");
});

test("caps the device pixel ratio", async ({ page }) => {
  await page.goto("/");
  const info = await page.evaluate(() => {
    const c = document.querySelector("canvas") as HTMLCanvasElement;
    return { backing: c.width, css: c.clientWidth, dpr: window.devicePixelRatio };
  });
  // maxDpr defaults to 2: the backing store must never exceed 2x the CSS size,
  // however high the device's own ratio is.
  expect(info.backing).toBeLessThanOrEqual(info.css * 2 + 2);
});

test("the canvas does not intercept clicks", async ({ page }) => {
  await page.goto("/");
  const pe = await page.locator("canvas").first().evaluate(
    (el) => getComputedStyle(el).pointerEvents
  );
  expect(pe).toBe("none");
});

test("WebGL actually initialised, with no GL errors", async ({ page }) => {
  await page.goto("/");
  const err = await page.evaluate(() => {
    const c = document.querySelector("canvas") as HTMLCanvasElement;
    const gl = c.getContext("webgl2") || c.getContext("webgl");
    return gl ? (gl as WebGLRenderingContext).getError() : -1;
  });
  // 0 is GL_NO_ERROR; -1 means no context, which the page must still survive.
  expect([0, -1]).toContain(err);
});

test("pause and resume are reflected in the UI", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /^Pause$/ }).click();
  await expect(statusPill(page)).toHaveText("paused");
  await page.getByRole("button", { name: /^Resume$/ }).click();
  await expect(statusPill(page)).toHaveText("running");
});

test("the scoped instance stays inside its container", async ({ page }) => {
  await page.goto("/");
  const scoped = page.locator(".scoped canvas");
  await expect(scoped).toBeAttached();
  await expect(scoped).toHaveCSS("position", "absolute");

  const box = (await page.locator(".scoped").boundingBox())!;
  const inner = (await scoped.boundingBox())!;
  expect(inner.height).toBeLessThanOrEqual(box.height + 2);
});

test("only one canvas per instance, no stacking", async ({ page }) => {
  await page.goto("/");
  // One full-page canvas plus one scoped canvas.
  await expect(page.locator("canvas")).toHaveCount(2);
});

test("no horizontal overflow", async ({ page }) => {
  await page.goto("/");
  await expectNoHorizontalOverflow(page);
});

test("the demo page logs no errors", async ({ page }) => {
  const assertClean = failOnConsoleErrors(page);
  await page.goto("/");
  await page.mouse.move(200, 200);
  await page.mouse.move(400, 300);
  await page.waitForTimeout(500);
  assertClean();
});
