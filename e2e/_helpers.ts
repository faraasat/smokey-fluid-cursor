import { expect, Page } from "@playwright/test";

/**
 * Fails when the page scrolls sideways.
 *
 * This is the check that would have caught the consent banner's missing
 * `flex-wrap`, and any future "one element refuses to shrink" regression.
 */
export async function expectNoHorizontalOverflow(page: Page) {
  const { doc, win, widest } = await page.evaluate(() => {
    let widest = { selector: "", width: 0 };
    for (const el of Array.from(document.body.querySelectorAll<HTMLElement>("*"))) {
      const r = el.getBoundingClientRect();
      if (r.right > widest.width) {
        widest = {
          selector:
            el.tagName.toLowerCase() +
            (el.className && typeof el.className === "string"
              ? "." + el.className.trim().split(/\s+/).slice(0, 2).join(".")
              : ""),
          width: Math.round(r.right),
        };
      }
    }
    return {
      doc: document.documentElement.scrollWidth,
      win: window.innerWidth,
      widest,
    };
  });

  expect(
    doc,
    `page scrolls sideways (${doc}px > ${win}px); widest element: ${widest.selector} ending at ${widest.width}px`
  ).toBeLessThanOrEqual(win + 1);
}

/** Fails when any text node is clipped by its own container. */
export async function expectNoClippedText(page: Page, selector: string) {
  const clipped = await page.evaluate((sel) => {
    const out: string[] = [];
    for (const el of Array.from(document.querySelectorAll<HTMLElement>(sel))) {
      // A deliberate line-clamp sets overflow hidden on purpose; only flag
      // elements that are not clamping.
      const style = getComputedStyle(el);
      if (style.webkitLineClamp && style.webkitLineClamp !== "none") continue;
      if (el.scrollWidth > el.clientWidth + 1) {
        out.push(`${el.className}: ${el.scrollWidth} > ${el.clientWidth}`);
      }
    }
    return out;
  }, selector);

  expect(clipped, `clipped text in ${selector}`).toEqual([]);
}

/** The demo pages must never log an error. */
export function failOnConsoleErrors(page: Page) {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(String(e)));
  return () => {
    // Analytics is blocked in CI (no network), which is not a demo bug.
    const real = errors.filter(
      (e) =>
        !/aptabase|googletagmanager|google-analytics|ERR_(INTERNET|NAME|BLOCKED|CONNECTION)|Failed to fetch|net::/i.test(
          e
        )
    );
    expect(real, "console errors on the demo page").toEqual([]);
  };
}
