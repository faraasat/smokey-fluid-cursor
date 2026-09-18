"use client";

import { useEffect, useRef, useState } from "react";
import { initFluid } from "smokey-fluid-cursor";
import { Hero } from "@/components/hero";
import { Footer } from "@/components/footer";
import { track } from "@/components/analytics";

export default function Home() {
  const [running, setRunning] = useState(true);
  const disposeRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!running) return;
    // initFluid returns a disposer: call it to stop the render loop and detach
    // the window listeners again.
    disposeRef.current = initFluid({ id: "demo-canvas" });
    return () => {
      disposeRef.current?.();
      disposeRef.current = null;
    };
  }, [running]);

  const toggle = () => {
    setRunning((r) => {
      track("simulation_toggled", { running: !r });
      return !r;
    });
  };

  return (
    <>
      <canvas id="demo-canvas" />

      <main className="wrap">
        <Hero />

        <section className="card">
          <h2>Live simulation</h2>
          <p className="sub">
            Move your pointer anywhere on the page. This demo is plain
            TypeScript — the package has no framework dependency.
          </p>
          <div className="row">
            <button className="demo primary" onClick={toggle}>
              {running ? "Stop simulation" : "Start simulation"}
            </button>
            <span className={`pill ${running ? "on" : "off"}`}>
              {running ? "running" : "stopped"}
            </span>
          </div>
        </section>

        <section className="card">
          <h2>Usage — ES modules</h2>
          <pre>{`import { initFluid } from "smokey-fluid-cursor";

const dispose = initFluid({ id: "my-canvas" });

// later, to tear it down again:
dispose();`}</pre>
        </section>

        <section className="card">
          <h2>Usage — script tag</h2>
          <p className="sub">
            An IIFE build is published for no-build pages —{" "}
            <a href="./vanilla.html">see it running in a single HTML file</a>.
          </p>
          <pre>{`<canvas id="my-canvas"></canvas>
<script src="https://unpkg.com/smokey-fluid-cursor"></script>
<script>
  SmokeyFluid.initFluid({ id: "my-canvas" });
</script>`}</pre>
        </section>

        <Footer />
      </main>
    </>
  );
}
