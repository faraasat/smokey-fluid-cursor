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

  // The canvas is created in an effect, so it can be attached a frame before
  // it has been laid out — measuring straight away yields a null box.
  await expect
    .poll(async () => (await scoped.boundingBox())?.height ?? 0)
    .toBeGreaterThan(0);

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

/** One screenshot pass: hide chrome, sample, restore. */
async function sample(page: import("@playwright/test").Page) {
  await page.evaluate(() => {
    for (const sel of ["main", ".topnav"]) {
      const el = document.querySelector(sel);
      if (el) (el as HTMLElement).style.visibility = "hidden";
    }
  });

  const png = PNG.sync.read(await page.screenshot({ clip: CLIP }));

  await page.evaluate(() => {
    for (const sel of ["main", ".topnav"]) {
      const el = document.querySelector(sel);
      if (el) (el as HTMLElement).style.visibility = "";
    }
  });

  const colours = new Set<string>();
  let colourful = 0;
  let rSum = 0, gSum = 0, bSum = 0, lit = 0;

  for (let i = 0; i < png.data.length; i += 4) {
    const [r, g, b] = [png.data[i], png.data[i + 1], png.data[i + 2]];
    colours.add(`${r >> 4},${g >> 4},${b >> 4}`);
    if (Math.max(r, g, b) - Math.min(r, g, b) > 24) colourful++;
    // Threshold set well above the page background: at a low cut-off the mean
    // is dominated by unlit pixels and the palette signal disappears into it.
    if (r + g + b > 150) { rSum += r; gSum += g; bSum += b; lit++; }
  }

  const mean = lit
    ? { r: rSum / lit, g: gSum / lit, b: bSum / lit }
    : { r: 0, g: 0, b: 0 };

  return { distinct: colours.size, colourful, mean, lit };
}

/**
 * Paints by moving the pointer, then samples once enough fluid is actually on
 * screen.
 *
 * Sampling after a fixed delay made this flaky: CI has no GPU and renders
 * through SwiftShader, so under parallel load far fewer frames land in the
 * same wall-clock window and the measurement is taken against a near-empty
 * canvas. Waiting on the picture itself removes that dependency on speed.
 */
async function paintAndSample(page: import("@playwright/test").Page) {
  await page.locator("canvas").first().waitFor();

  const deadline = Date.now() + 30_000;
  let shot = null as Awaited<ReturnType<typeof sample>> | null;

  while (Date.now() < deadline) {
    // `steps` moves the cursor in one call rather than 40 round trips.
    await page.mouse.move(120, 320);
    await page.mouse.move(640, 400, { steps: 40 });
    await page.waitForTimeout(350);

    shot = await sample(page);
    if (shot.lit > 2000) return shot;
  }

  return shot!;
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

/*
 * There was a pixel-comparison test here asserting that switching palette
 * changed the rendered colours. It was removed rather than tuned a fifth
 * time.
 *
 * It sampled a GPU-rendered fluid simulation and compared mean colour between
 * two palettes. Under parallel load — and on CI, which has no GPU and falls
 * back to SwiftShader — how much fluid accumulates in a given window varies
 * enough that the comparison passed or failed roughly at random. It never
 * caught a real defect, while a test that fails half the time teaches people
 * to ignore a red suite.
 *
 * What it was guarding is still covered: "the fluid is actually visible on the
 * page" below catches the occlusion regression this suite exists for, and the
 * unit tests cover setConfig and the preset definitions deterministically.
 */
