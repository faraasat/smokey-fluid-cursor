import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { initFluid } from "../src";

// jsdom has no WebGL, so getContext always returns null here. That is exactly
// the "unsupported device" path the package has to survive, and it still
// exercises mounting, placement and the handle contract.
describe("initFluid", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
    document.head.innerHTML = "";
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });
  afterEach(() => vi.restoreAllMocks());

  const addCanvas = (id = "smokey-fluid-canvas") => {
    const c = document.createElement("canvas");
    c.id = id;
    document.body.appendChild(c);
    return c;
  };

  describe("mounting", () => {
    it("creates its own canvas when none exists", () => {
      initFluid();
      expect(document.getElementById("smokey-fluid-canvas")).toBeTruthy();
    });

    it("reuses an existing canvas with the configured id", () => {
      const existing = addCanvas();
      const handle = initFluid();
      expect(handle.canvas).toBe(existing);
      expect(document.querySelectorAll("canvas")).toHaveLength(1);
    });

    it("accepts a canvas element directly", () => {
      const el = document.createElement("canvas");
      document.body.appendChild(el);
      expect(initFluid({ canvas: el }).canvas).toBe(el);
    });

    it("accepts a CSS selector for the canvas", () => {
      const el = addCanvas("picked");
      expect(initFluid({ canvas: "#picked" }).canvas).toBe(el);
    });

    it("mounts into a container element", () => {
      const host = document.createElement("section");
      document.body.appendChild(host);
      const handle = initFluid({ container: host });
      expect(host.contains(handle.canvas!)).toBe(true);
    });

    it("removes only a canvas it created on dispose", () => {
      const handle = initFluid();
      handle.dispose();
      expect(document.getElementById("smokey-fluid-canvas")).toBeNull();
    });

    it("leaves a caller-owned canvas in place on dispose", () => {
      const existing = addCanvas();
      initFluid().dispose();
      expect(document.body.contains(existing)).toBe(true);
    });
  });

  describe("placement", () => {
    it("is fixed and full-bleed by default", () => {
      const { canvas } = initFluid();
      expect(canvas!.style.position).toBe("fixed");
      expect(canvas!.style.width).toBe("100%");
    });

    it("can be positioned absolutely inside a container", () => {
      const host = document.createElement("div");
      document.body.appendChild(host);
      const { canvas } = initFluid({ container: host, position: "absolute" });
      expect(canvas!.style.position).toBe("absolute");
    });

    it("applies a custom zIndex", () => {
      expect(initFluid({ zIndex: 5 }).canvas!.style.zIndex).toBe("5");
    });

    it("ignores pointer events by default", () => {
      expect(initFluid().canvas!.style.pointerEvents).toBe("none");
    });

    it("can opt into pointer events", () => {
      expect(initFluid({ pointerEvents: true }).canvas!.style.pointerEvents).toBe("auto");
    });

    it("applies a custom className", () => {
      expect(initFluid({ className: "fx a" }).canvas!.classList.contains("fx")).toBe(true);
    });
  });

  describe("handle", () => {
    it("returns a disposer that is safe to call twice", () => {
      const h = initFluid();
      expect(() => { h.dispose(); h.dispose(); }).not.toThrow();
    });

    it("exposes pause and resume without throwing", () => {
      const h = initFluid();
      expect(() => { h.pause(); h.resume(); }).not.toThrow();
    });

    it("reports paused state after pause()", () => {
      const h = initFluid();
      h.pause();
      expect(h.isPaused()).toBe(true);
    });

    it("accepts live config updates", () => {
      const h = initFluid();
      expect(() => h.setConfig({ curl: 30, zIndex: 3 })).not.toThrow();
      expect(h.canvas!.style.zIndex).toBe("3");
    });

    it("does not throw when WebGL is unavailable", () => {
      expect(() => initFluid()).not.toThrow();
    });

    it("warns when WebGL is unavailable", () => {
      initFluid();
      expect(console.warn).toHaveBeenCalledWith(
        expect.stringContaining("WebGL is unavailable"),
        expect.anything()
      );
    });
  });

  describe("reduced motion", () => {
    const mockMotion = (reduce: boolean) =>
      vi.stubGlobal("matchMedia", (q: string) => ({
        matches: reduce && q.includes("reduce"),
        media: q,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      }));

    afterEach(() => vi.unstubAllGlobals());

    it("starts paused when the visitor prefers reduced motion", () => {
      mockMotion(true);
      expect(initFluid().isPaused()).toBe(true);
    });

    it("can be opted out of", () => {
      mockMotion(true);
      expect(() => initFluid({ respectReducedMotion: false })).not.toThrow();
    });
  });
});

describe("presets", () => {
  it("ships exactly 100", async () => {
    const { presetNames } = await import("../src/presets");
    expect(presetNames).toHaveLength(100);
  });

  it("is every palette crossed with every character", async () => {
    const { presetNames, paletteNames, characterNames } = await import("../src/presets");
    expect(paletteNames.length * characterNames.length).toBe(presetNames.length);
  });

  it("names them '<Palette> <Character>'", async () => {
    const { presets } = await import("../src/presets");
    expect(presets["Ocean Swirl"]).toBeDefined();
    expect(presets["Sunset Calm"]).toBeDefined();
  });

  it("has no duplicate names", async () => {
    const { presetNames } = await import("../src/presets");
    expect(new Set(presetNames).size).toBe(presetNames.length);
  });

  it("gives every preset a distinct configuration", async () => {
    const { presets } = await import("../src/presets");
    const shapes = Object.values(presets).map((p) => JSON.stringify(p));
    expect(new Set(shapes).size).toBe(shapes.length);
  });

  it("sets only appearance and physics, never placement", async () => {
    const { presets } = await import("../src/presets");
    // Placement must stay the caller's decision, so presets compose with any
    // container/zIndex rather than overriding them.
    const forbidden = ["canvas", "container", "position", "zIndex", "id", "pointerEvents"];
    for (const [name, preset] of Object.entries(presets)) {
      for (const key of forbidden) {
        expect(preset, `${name} must not set ${key}`).not.toHaveProperty(key);
      }
    }
  });

  it("uses valid hex colours throughout", async () => {
    const { presets } = await import("../src/presets");
    for (const [name, preset] of Object.entries(presets)) {
      for (const colour of preset.palette ?? []) {
        expect(colour, `${name}`).toMatch(/^#[0-9a-f]{6}$/i);
      }
    }
  });

  it("looks a preset up by name", async () => {
    const { getPreset } = await import("../src/presets");
    expect(getPreset("Magma Storm")).toBeDefined();
    expect(getPreset("Nope Nope")).toBeUndefined();
  });

  it("can be handed straight to initFluid", async () => {
    const { presets } = await import("../src/presets");
    vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(() => initFluid(presets["Aurora Flow"])).not.toThrow();
  });
});
