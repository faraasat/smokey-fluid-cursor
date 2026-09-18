import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { initFluid } from "../src";

describe("initFluid", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
    document.head.innerHTML = "";
  });
  afterEach(() => vi.restoreAllMocks());

  const addCanvas = (id = "smokey-fluid-canvas") => {
    const c = document.createElement("canvas");
    c.id = id;
    document.body.appendChild(c);
    return c;
  };

  it("returns a disposer even when the canvas is missing", () => {
    const dispose = initFluid({});
    expect(typeof dispose).toBe("function");
    expect(() => dispose()).not.toThrow();
  });

  it("does nothing when no canvas matches the configured id", () => {
    addCanvas("some-other-id");
    expect(() => initFluid({})).not.toThrow();
    expect(document.head.querySelector("style")).toBeNull();
  });

  it("does not throw when WebGL is unavailable", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    addCanvas();
    expect(() => initFluid({})).not.toThrow();
  });

  it("warns when WebGL is unavailable", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    addCanvas();
    initFluid({});
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining("WebGL is unavailable"),
      expect.anything()
    );
  });

  it("cleans up its injected style when WebGL is unavailable", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    addCanvas();
    initFluid({});
    expect(document.head.querySelector("style")).toBeNull();
  });

  it("targets a custom canvas id from config", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    addCanvas("custom");
    const warn = vi.spyOn(console, "warn");
    initFluid({ id: "custom" });
    expect(warn).toHaveBeenCalled(); // reached the WebGL stage, so the canvas was found
  });

  it("returns a disposer that is safe to call twice", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    addCanvas();
    const dispose = initFluid({});
    expect(() => { dispose(); dispose(); }).not.toThrow();
  });
});
