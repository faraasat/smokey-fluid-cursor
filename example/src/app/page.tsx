"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { initFluid, presets, presetNames, paletteNames, characterNames } from "smokey-fluid-cursor";
import type { FluidHandle, Preset, PresetName } from "smokey-fluid-cursor";
import { Hero } from "@/components/hero";
import { Footer } from "@/components/footer";
import { Code } from "@/components/code";
import { track } from "@/components/analytics";

export default function Home() {
  const fluid = useRef<FluidHandle | null>(null);
  const [paused, setPaused] = useState(false);
  const [preset, setPreset] = useState<PresetName>("Spectrum Flow");
  const [palette, setPalette] = useState<string>("");
  const [query, setQuery] = useState("");

  // Overrides layered on top of the chosen preset.
  const [curl, setCurl] = useState<number | null>(null);
  const [intensity, setIntensity] = useState<number | null>(null);

  /** The configuration currently driving both simulations on this page. */
  const active: Preset = useMemo(() => {
    const base = presets[preset];
    return {
      ...base,
      ...(curl !== null ? { curl } : {}),
      ...(intensity !== null ? { colorIntensity: intensity } : {}),
    };
  }, [preset, curl, intensity]);

  useEffect(() => {
    const handle = initFluid({ id: "demo-canvas", ...active });
    fluid.current = handle;
    setPaused(handle.isPaused());
    return () => handle.dispose();
    // Re-created only when the preset changes; cheap tweaks go through
    // setConfig below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset]);

  // Slider changes are pushed into the running simulation, not remounted.
  useEffect(() => {
    fluid.current?.setConfig(active);
  }, [active]);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? presetNames.filter((n) => n.toLowerCase().includes(q)) : presetNames;
  }, [query]);

  const choose = (name: PresetName) => {
    setPreset(name);
    setCurl(null);
    setIntensity(null);
    track("preset_selected", { preset: name });
  };

  const toggle = () => {
    const h = fluid.current;
    if (!h) return;
    h.isPaused() ? h.resume() : h.pause();
    setPaused(h.isPaused());
  };

  return (
    <main className="wrap">
      <Hero />

      <section className="card">
        <h2>Live controls</h2>
        <p className="sub">
          Move your pointer anywhere on the page. Everything below retunes the
          running simulation through <code>setConfig()</code> — nothing is
          remounted.
        </p>

        <div className="row">
          <button className="demo primary" onClick={toggle}>
            {paused ? "Resume" : "Pause"}
          </button>
          <span className={`pill ${paused ? "off" : "on"}`}>
            {paused ? "paused" : "running"}
          </span>
          <span className="pill">{preset}</span>
        </div>

        <div className="field">
          <label htmlFor="curl">
            Swirl <code>curl: {curl ?? active.curl}</code>
          </label>
          <input
            id="curl"
            type="range"
            min={0}
            max={50}
            value={curl ?? active.curl ?? 10}
            onChange={(e) => setCurl(Number(e.target.value))}
          />
        </div>

        <div className="field">
          <label htmlFor="intensity">
            Brightness{" "}
            <code>colorIntensity: {(intensity ?? active.colorIntensity ?? 0.15).toFixed(2)}</code>
          </label>
          <input
            id="intensity"
            type="range"
            min={0.05}
            max={0.6}
            step={0.05}
            value={intensity ?? active.colorIntensity ?? 0.15}
            onChange={(e) => setIntensity(Number(e.target.value))}
          />
        </div>
      </section>

      <section className="card">
        <h2>100 presets</h2>
        <p className="sub">
          Every palette crossed with every motion character, shipped in the
          package as <code>presets</code>. {paletteNames.length} palettes ×{" "}
          {characterNames.length} characters.
        </p>

        <div className="preset-filter">
          <input
            type="search"
            className="preset-search"
            placeholder={`Filter ${presetNames.length} presets…`}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Filter presets"
          />
          <div className="row">
            {paletteNames.slice(0, 6).map((p) => (
              <button
                key={p}
                className={`demo${palette === p ? " primary" : ""}`}
                onClick={() => {
                  const next = palette === p ? "" : p;
                  setPalette(next);
                  setQuery(next);
                }}
              >
                {p}
              </button>
            ))}
          </div>
        </div>

        <p className="sub" style={{ marginTop: 14 }}>
          Showing {visible.length} of {presetNames.length}
        </p>

        <ul className="presets">
          {visible.map((name) => (
            <li key={name}>
              <button
                className={`preset${preset === name ? " is-active" : ""}`}
                onClick={() => choose(name)}
                aria-pressed={preset === name}
              >
                <span className="preset__swatch" aria-hidden="true">
                  {(presets[name].palette ?? ["#ff4ecd", "#4ea8ff", "#ffd24e"]).map((c, i) => (
                    <i key={i} style={{ background: c }} />
                  ))}
                </span>
                <span className="preset__name">{name}</span>
              </button>
            </li>
          ))}
        </ul>
      </section>

      <section className="card">
        <h2>Scoped to a container</h2>
        <p className="sub">
          With <code>position: &quot;absolute&quot;</code> and a{" "}
          <code>container</code>, the effect stays inside one element. This one
          follows the controls above, so you can compare the same settings at
          two scales.
        </p>
        <ScopedDemo config={active} />
      </section>

      <section className="card">
        <h2>Usage</h2>
        <Code language="tsx">{`import { initFluid, presets } from "smokey-fluid-cursor";

const fluid = initFluid(presets["Ocean Swirl"]);

fluid.pause();
fluid.setConfig({ curl: 30 });
fluid.dispose();`}</Code>
      </section>

      <section className="card">
        <h2>Without a bundler</h2>
        <p className="sub">
          A minified IIFE build ships for no-build pages —{" "}
          <a href="./vanilla.html">see it running in a single HTML file</a>.
        </p>
        <Code language="html">{`<script src="https://unpkg.com/smokey-fluid-cursor"></script>
<script>
  var fluid = SmokeyFluid.initFluid(SmokeyFluid.presets["Magma Storm"]);
</script>`}</Code>
      </section>

      <Footer />
    </main>
  );
}

/** A second simulation, scoped to its own box, driven by the same config. */
function ScopedDemo({ config }: { config: Preset }) {
  const boxRef = useRef<HTMLDivElement>(null);
  const handle = useRef<FluidHandle | null>(null);

  useEffect(() => {
    if (!boxRef.current) return;
    const h = initFluid({
      id: "scoped-canvas",
      container: boxRef.current,
      position: "absolute",
      zIndex: 0,
      // Cheaper: this canvas is a fraction of the viewport.
      dyeResolution: 512,
      ...config,
    });
    handle.current = h;
    return () => h.dispose();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keep it in step with the controls above rather than running its own look.
  useEffect(() => {
    handle.current?.setConfig(config);
  }, [config]);

  return (
    <div ref={boxRef} className="scoped">
      <span>Move your pointer in here</span>
    </div>
  );
}
