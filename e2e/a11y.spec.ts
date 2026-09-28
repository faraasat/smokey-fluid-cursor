import { test, expect, Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

/**
 * Waits for CSS animations to finish before auditing.
 *
 * Measuring mid-fade blends foreground against backdrop and reports contrast
 * failures that do not exist at rest.
 */
async function settled(page: Page) {
  await page.waitForFunction(() =>
    document.getAnimations().every((a) => a.playState === "finished")
  );
}

const audit = (page: Page) =>
  new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();

const describe = (violations: Awaited<ReturnType<typeof audit>>["violations"]) =>
  violations.flatMap((v) =>
    v.nodes.map(
      (n) =>
        `${v.id} (${v.impact}) at ${n.target} — ${n.failureSummary?.replace(/\s+/g, " ")}`
    )
  );

test("demo page has no accessibility violations", async ({ page }) => {
  await page.goto("/");
  await page.locator(".card").first().waitFor();
  await settled(page);

  const { violations } = await audit(page);
  expect(describe(violations)).toEqual([]);
});

