"use client";

import { useEffect, useRef, useState } from "react";
import { initFluid } from "smokey-fluid-cursor";
import type { FluidHandle } from "smokey-fluid-cursor";
import { Hero } from "@/components/hero";
import { Footer } from "@/components/footer";
import { track } from "@/components/analytics";

const PALETTES: Record<string, string[] | null> = {
  Spectrum: null,
  Sunset: ["#ff4ecd", "#ff8a4e", "#ffd24e"],
  Ocean: ["#4ea8ff", "#4effd2", "#7c4dff"],
  Mono: ["#ffffff"],
};

export default function Home() {
  const handleRef = useRef<FluidHandle | null>(null);
  const [paused, setPaused] = useState(false);
  const [palette, setPalette] = useState("Spectrum");
  const [curl, setCurl] = useState(10);
  const [intensity, setIntensity] = useState(0.15);

  useEffect(() => {
    // initFluid creates its own canvas when none exists, and returns a handle
    // that tears everything down again.
    const handle = initFluid({ id: "demo-canvas" });
    handleRef.current = handle;
    setPaused(handle.isPaused());
    return () => handle.dispose();
  }, []);

  const apply = (patch: Parameters<FluidHandle["setConfig"]>[0]) =>
    handleRef.current?.setConfig(patch);

  const toggle = () => {
    const h = handleRef.current;
    if (!h) return;
    if (h.isPaused()) h.resume();
    else h.pause();
    setPaused(h.isPaused());
    track("simulation_toggled", { paused: h.isPaused() });
  };

  return (
    <main className="wrap">
      <Hero />

      <section className="card">
        <h2>Live controls</h2>
        <p className="sub">
          Move your pointer anywhere on the page. Every control below calls{" "}
          <code>setConfig()</code> on the running simulation — nothing is
          remounted.
        </p>

        <div className="row">
          <button className="demo primary" onClick={toggle}>
            {paused ? "Resume" : "Pause"}
          </button>
          <span className={`pill ${paused ? "off" : "on"}`}>
            {paused ? "paused" : "running"}
          </span>
        </div>

        <div className="field">
          <label htmlFor="palette">Palette</label>
          <div className="row" id="palette">
            {Object.keys(PALETTES).map((name) => (
              <button
                key={name}
                className={`demo${palette === name ? " primary" : ""}`}
                onClick={() => {
                  setPalette(name);
                  apply({ palette: PALETTES[name] });
                  track("palette_changed", { palette: name });
                }}
              >
                {name}
              </button>
            ))}
          </div>
        </div>

        <div className="field">
          <label htmlFor="curl">
            Swirl <code>curl: {curl}</code>
          </label>
          <input
            id="curl"
            type="range"
            min={0}
            max={50}
            value={curl}
            onChange={(e) => {
              const v = Number(e.target.value);
              setCurl(v);
              apply({ curl: v });
            }}
          />
        </div>

        <div className="field">
          <label htmlFor="intensity">
            Brightness <code>colorIntensity: {intensity.toFixed(2)}</code>
          </label>
          <input
            id="intensity"
            type="range"
            min={0.05}
            max={0.6}
            step={0.05}
            value={intensity}
            onChange={(e) => {
              const v = Number(e.target.value);
              setIntensity(v);
              apply({ colorIntensity: v });
            }}
          />
        </div>
      </section>

      <section className="card">
        <h2>Scoped to a container</h2>
        <p className="sub">
          With <code>position: &quot;absolute&quot;</code> and a{" "}
          <code>container</code>, the effect stays inside one element instead of
          covering the viewport.
        </p>
        <ScopedDemo />
      </section>

      <section className="card">
        <h2>Usage</h2>
        <pre>{`import { initFluid } from "smokey-fluid-cursor";

const fluid = initFluid();

fluid.pause();
fluid.setConfig({ curl: 30, palette: ["#ff4ecd"] });
fluid.dispose();`}</pre>
      </section>

      <section className="card">
        <h2>Without a bundler</h2>
        <p className="sub">
          A minified IIFE build ships for no-build pages —{" "}
          <a href="./vanilla.html">see it running in a single HTML file</a>.
        </p>
        <pre>{`<script src="https://unpkg.com/smokey-fluid-cursor"></script>
<script>
  var fluid = SmokeyFluid.initFluid();
</script>`}</pre>
      </section>

      <Footer />
    </main>
  );
}

function ScopedDemo() {
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!boxRef.current) return;
    const handle = initFluid({
      id: "scoped-canvas",
      container: boxRef.current,
      position: "absolute",
      zIndex: 0,
      palette: ["#4ea8ff", "#7c4dff"],
      dyeResolution: 512,
    });
    return () => handle.dispose();
  }, []);

  return (
    <div ref={boxRef} className="scoped">
      <span>Move your pointer in here</span>
    </div>
  );
}
