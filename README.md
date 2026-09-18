<p align="center">
  <img src="https://raw.githubusercontent.com/faraasat/smokey-fluid-cursor/main/.github/assets/banner.svg" alt="smokey-fluid-cursor" width="100%" />
</p>

<p align="center">
  A GPU-accelerated fluid-simulation cursor trail for any website. No framework, no dependencies.
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/smokey-fluid-cursor"><img alt="npm version" src="https://img.shields.io/npm/v/smokey-fluid-cursor?color=cb3837&label=npm&logo=npm"></a>
  <a href="https://www.npmjs.com/package/smokey-fluid-cursor"><img alt="downloads" src="https://img.shields.io/npm/dm/smokey-fluid-cursor?color=cb3837&label=downloads"></a>
  <a href="https://bundlephobia.com/package/smokey-fluid-cursor"><img alt="bundle size" src="https://img.shields.io/bundlephobia/minzip/smokey-fluid-cursor?label=minzipped"></a>
  <a href="https://github.com/faraasat/smokey-fluid-cursor/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/faraasat/smokey-fluid-cursor/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="types" src="https://img.shields.io/badge/types-included-3178c6?logo=typescript&logoColor=white">
  <a href="https://github.com/faraasat/smokey-fluid-cursor/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/npm/l/smokey-fluid-cursor?color=blue"></a>
</p>

<p align="center">
  <a href="https://faraasat.github.io/smokey-fluid-cursor/"><b>Live demo</b></a> ·
  <a href="https://www.npmjs.com/package/smokey-fluid-cursor">npm</a> ·
  <a href="https://github.com/faraasat/smokey-fluid-cursor/blob/main/CHANGELOG.md">Changelog</a> ·
  <a href="https://github.com/faraasat/smokey-fluid-cursor/issues">Issues</a>
</p>

---

## Why

A real-time Navier–Stokes fluid solver running in WebGL, wired to your pointer.
It ships as ESM, CJS and a plain `<script>` IIFE build, so it drops into a
bundled app or a single HTML file equally well — and it degrades quietly on
devices without WebGL instead of taking the page down.

> Using React or Next.js? See
> [`react-smokey-fluid-cursor`](https://github.com/faraasat/react-smokey-fluid-cursor).

## Installation

```bash
npm install smokey-fluid-cursor
```

<details>
<summary>yarn / pnpm / bun</summary>

```bash
yarn add smokey-fluid-cursor
pnpm add smokey-fluid-cursor
bun add smokey-fluid-cursor
```
</details>

No peer dependencies.

## Quick start

```html
<canvas id="smokey-fluid-canvas"></canvas>
```

```ts
import { initFluid } from "smokey-fluid-cursor";

const dispose = initFluid();
```

`initFluid` returns a **disposer**. Call it to stop the render loop and detach
every window listener — required in any single-page app, or each navigation
leaks a whole simulation:

```ts
dispose();
```

The canvas is positioned `fixed`, full-viewport, `pointer-events: none` and
`z-index: -9999`, so it sits behind your content and never intercepts clicks.

## Without a bundler

```html
<canvas id="smokey-fluid-canvas"></canvas>
<script src="https://unpkg.com/smokey-fluid-cursor"></script>
<script>
  var dispose = SmokeyFluid.initFluid();
</script>
```

## Configuration

Every field is optional.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `id` | `string` | `"smokey-fluid-canvas"` | Id of the canvas to render into. |
| `simResolution` | `number` | `128` | Velocity/pressure grid. Lower is faster and coarser. |
| `dyeResolution` | `number` | `1440` | Colour buffer resolution. The main quality/cost dial. |
| `densityDissipation` | `number` | `3.5` | How fast colour fades. Higher fades sooner. |
| `velocityDissipation` | `number` | `2` | How fast motion slows. |
| `pressure` | `number` | `0.1` | Initial pressure multiplier. |
| `pressureIteration` | `number` | `20` | Jacobi iterations. Higher is more accurate, slower. |
| `curl` | `number` | `10` | Vorticity confinement — the swirliness. |
| `splatRadius` | `number` | `0.5` | Size of each pointer splat. |
| `splatForce` | `number` | `6000` | Force applied per splat. |
| `shading` | `boolean` | `true` | Lighting for a sense of depth. |
| `colorUpdateSpeed` | `number` | `10` | How fast the palette rotates. |
| `backColor` | `{ r, g, b }` | `{ r: 0, g: 0, b: 0 }` | Canvas background. |
| `transparent` | `boolean` | `true` | Blend with the page background. |
| `paused` | `boolean` | `false` | Freeze the simulation. |

```ts
initFluid({
  curl: 30,
  splatForce: 9000,
  densityDissipation: 2,
  id: "my-canvas",
});
```

## Performance

The defaults target a modern desktop GPU. On lower-powered devices, drop
`dyeResolution` to `512` and `pressureIteration` to `10`. The simulation
automatically lowers quality when the GPU lacks linear filtering for float
textures.

## Browser support

Requires WebGL (WebGL 2 is used when available, with a WebGL 1 fallback). On a
device without it, `initFluid` logs a warning, renders nothing, and returns a
no-op disposer — it never throws.

## Contributing

Issues and pull requests are welcome.

```bash
git clone https://github.com/faraasat/smokey-fluid-cursor.git
cd smokey-fluid-cursor
npm install
npm test          # vitest
npm run typecheck # tsc --noEmit
npm run build     # tsup
```

To run the demo site against your local build:

```bash
npm run example:dev
```

Releases are manual — nothing publishes on a push to `main`. Maintainers run
the **Release** workflow from the Actions tab.

## Privacy

The published package contains **no telemetry**. The demo site at
[faraasat.github.io/smokey-fluid-cursor](https://faraasat.github.io/smokey-fluid-cursor/) uses
Google Analytics and Aptabase; the library itself never phones home.

## License

[MIT](./LICENSE) © [Farasat Ali](https://github.com/faraasat)
