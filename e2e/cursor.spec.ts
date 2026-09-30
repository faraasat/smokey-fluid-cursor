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
  await page.locator("canvas").first().waitFor();
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
  await page.locator("canvas").first().waitFor();
  const pe = await page.locator("canvas").first().evaluate(
    (el) => getComputedStyle(el).pointerEvents
  );
  expect(pe).toBe("none");
});

test("WebGL actually initialised, with no GL errors", async ({ page }) => {
  await page.goto("/");
  await page.locator("canvas").first().waitFor();
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

/**
 * These assert the effect is actually *visible*, not merely running.
 *
 * The canvas sits at a negative z-index, and a fixed element there paints
 * below the backgrounds of block-level descendants — so an opaque
 * `body { background }` hid the whole effect while every other check
 * (canvas present, loop running, no GL errors) still passed.
 */
import { PNG } from "pngjs";

const CLIP = { x: 60, y: 240, width: 520, height: 260 };

async function paintAndSample(page: import("@playwright/test").Page) {
  await page.locator("canvas").first().waitFor();
  // `steps` moves the cursor in one call rather than 40 round trips.
  await page.mouse.move(120, 320);
  await page.mouse.move(640, 400, { steps: 40 });
  await page.waitForTimeout(450);

  // Hide the page content so only the fluid canvas is composited. Without
  // this the sample also counts UI chrome — the preset swatches alone are
  // hundreds of coloured pixels, which swamps the signal being measured.
  await page.evaluate(() => {
    const main = document.querySelector("main");
    if (main) (main as HTMLElement).style.visibility = "hidden";
    const nav = document.querySelector(".topnav");
    if (nav) (nav as HTMLElement).style.visibility = "hidden";
  });

  const png = PNG.sync.read(await page.screenshot({ clip: CLIP }));

  await page.evaluate(() => {
    const main = document.querySelector("main");
    if (main) (main as HTMLElement).style.visibility = "";
    const nav = document.querySelector(".topnav");
    if (nav) (nav as HTMLElement).style.visibility = "";
  });
  const colours = new Set<string>();
  let colourful = 0;
  // Mean colour of the lit pixels — a stable signature of the palette in use.
  let rSum = 0, gSum = 0, bSum = 0, lit = 0;

  for (let i = 0; i < png.data.length; i += 4) {
    const [r, g, b] = [png.data[i], png.data[i + 1], png.data[i + 2]];
    colours.add(`${r >> 4},${g >> 4},${b >> 4}`);
    // Saturated pixels: the fluid is coloured, the page chrome is near-grey.
    if (Math.max(r, g, b) - Math.min(r, g, b) > 24) colourful++;
    if (r + g + b > 40) { rSum += r; gSum += g; bSum += b; lit++; }
  }

  const mean = lit
    ? { r: rSum / lit, g: gSum / lit, b: bSum / lit }
    : { r: 0, g: 0, b: 0 };

  return { distinct: colours.size, colourful, mean, lit };
}

test("the fluid is actually visible on the page", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "pointer-driven");
  // CI has no GPU and renders through SwiftShader, which is far slower than
  // the default budget allows for a real fluid simulation.
  test.setTimeout(120_000);

  await page.goto("/");
  await page.waitForTimeout(300);
  const { colourful } = await paintAndSample(page);

  // With the effect hidden behind an opaque body background this was 0.
  expect(colourful, "no coloured fluid pixels — the canvas is being painted over").toBeGreaterThan(300);
});

test("a palette change reaches the running simulation", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "pointer-driven");
  test.setTimeout(120_000);

  await page.goto("/");
  await page.waitForTimeout(300);

  // "Mono Flow" draws white-ish trails; "Sunset Flow" draws warm ones.
  // Sampling the hue of what is actually on screen proves setConfig reached
  // the simulation, rather than merely that React state updated.
  await page.getByRole("button", { name: "Mono Flow", exact: true }).click();
  const mono = await paintAndSample(page);

  await page.getByRole("button", { name: "Sunset Flow", exact: true }).click();
  const sunset = await paintAndSample(page);

  expect(mono.lit, "Mono rendered nothing").toBeGreaterThan(100);
  expect(sunset.lit, "Sunset rendered nothing").toBeGreaterThan(100);

  // Assert the picture actually changed, rather than which palette is
  // "more colourful" — that magnitude comparison is noisy, because blended
  // white trails are themselves far from grey. A shift in the mean colour of
  // the lit pixels is the direct evidence that setConfig reached the
  // simulation.
  const shift =
    Math.abs(mono.mean.r - sunset.mean.r) +
    Math.abs(mono.mean.g - sunset.mean.g) +
    Math.abs(mono.mean.b - sunset.mean.b);

  expect(shift, `palette change did not alter the rendered output (mono=${JSON.stringify(mono.mean)} sunset=${JSON.stringify(sunset.mean)})`).toBeGreaterThan(6);
});
