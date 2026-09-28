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

```ts
import { initFluid } from "smokey-fluid-cursor";

const fluid = initFluid();
```

That is the whole integration. With no options it creates a full-viewport
canvas in `<body>`, positioned `fixed`, `pointer-events: none` and behind your
content, then starts the simulation.

`initFluid` returns a **handle**. Keep it if you need to stop, pause or retune
the effect later:

```ts
fluid.pause();
fluid.resume();
fluid.setConfig({ curl: 30 });
fluid.dispose(); // stops the loop and detaches every listener
```

Calling `dispose()` is required in any single-page app — otherwise each
navigation leaks a whole simulation.

### Bring your own canvas

```html
<canvas id="my-canvas"></canvas>
```

```ts
initFluid({ canvas: "#my-canvas" });
```

### Scope it to one section

Give it a container and switch to `absolute`, and the effect stays inside that
element instead of covering the page:

```ts
initFluid({
  container: "#hero",   // element or selector
  position: "absolute",
  zIndex: 0,
});
```

The container needs its own positioning context (`position: relative`) and
`overflow: hidden` if you want the fluid clipped to it.

## Without a bundler

A **minified** IIFE build is published for no-build pages:

```html
<script src="https://unpkg.com/smokey-fluid-cursor"></script>
<script>
  var fluid = SmokeyFluid.initFluid();
</script>
```

## The handle

| Method | Description |
| --- | --- |
| `dispose()` | Stop the loop, detach listeners, release the GL context, remove any canvas this call created. Safe to call twice. |
| `pause()` | Freeze the simulation, leaving the canvas visible. |
| `resume()` | Resume after `pause()`. |
| `isPaused()` | Whether the simulation is currently stopped. |
| `setConfig(partial)` | Retune in place — no remount. Resolution changes reallocate framebuffers; everything else applies next frame. |
| `splat(x, y, color?)` | Inject a splash at a point, in CSS pixels relative to the canvas. Drive the effect from something other than the pointer. |
| `canvas` | The canvas being rendered into. |

## Configuration

Every option is optional.

### Mounting & placement

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `canvas` | `HTMLCanvasElement \| string` | — | Render into an existing canvas (element or selector). Wins over `id`/`container`. |
| `container` | `HTMLElement \| string` | `document.body` | Where to create the canvas when none exists. |
| `id` | `string` | `"smokey-fluid-canvas"` | Id used to find, or assign to, the canvas. |
| `position` | `"fixed" \| "absolute" \| "relative" \| "static"` | `"fixed"` | `absolute` confines the effect to a positioned container. |
| `zIndex` | `number` | `-9999` | Stacking order. |
| `pointerEvents` | `boolean` | `false` | Whether the canvas swallows clicks. |
| `className` | `string` | — | Extra class on the canvas. |

### Performance & accessibility

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `maxDpr` | `number` | `2` | Caps the device pixel ratio. Uncapped, a 3x phone renders **nine times** the pixels of a 1x display for a decorative effect. |
| `pauseOnHidden` | `boolean` | `true` | Stop the loop while the tab is in the background. |
| `respectReducedMotion` | `boolean` | `true` | Start paused when the visitor has `prefers-reduced-motion: reduce`. |
| `paused` | `boolean` | `false` | Start frozen. |

### Appearance

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `palette` | `string[]` | `null` | Hex colours to draw from, e.g. `["#ff4ecd", "#4ea8ff"]`. Omit for the full random hue range. |
| `colorIntensity` | `number` | `0.15` | Brightness multiplier. Raise for a bolder trail. |
| `backColor` | `{ r, g, b }` | `{ r: 0, g: 0, b: 0 }` | Canvas background. |
| `transparent` | `boolean` | `true` | Blend with the page background. |
| `shading` | `boolean` | `true` | Lighting, for a sense of depth. |
| `colorUpdateSpeed` | `number` | `10` | How fast the palette rotates. |

### Simulation

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `simResolution` | `number` | `128` | Velocity/pressure grid. Lower is faster and coarser. |
| `dyeResolution` | `number` | `1440` | Colour buffer resolution. The main quality/cost dial. |
| `densityDissipation` | `number` | `3.5` | How fast colour fades. |
| `velocityDissipation` | `number` | `2` | How fast motion slows. |
| `pressure` | `number` | `0.1` | Initial pressure multiplier. |
| `pressureIteration` | `number` | `20` | Jacobi iterations. Higher is more accurate, slower. |
| `curl` | `number` | `10` | Vorticity confinement — the swirliness. |
| `splatRadius` | `number` | `0.5` | Size of each pointer splat. |
| `splatForce` | `number` | `6000` | Force applied per splat. |

## Performance

The defaults target a modern desktop GPU. The big levers, in order of impact:

1. **`maxDpr`** — already capped at `2`. Drop to `1` for the weakest devices.
2. **`dyeResolution`** — `512` is noticeably cheaper and still looks good.
3. **`pressureIteration`** — `10` roughly halves the solver cost.

```ts
initFluid({ maxDpr: 1, dyeResolution: 512, pressureIteration: 10 });
```

Quality is also lowered automatically when the GPU lacks linear filtering for
float textures.

## Accessibility

A full-screen animation is a real problem for people with vestibular
disorders. By default this package honours `prefers-reduced-motion: reduce` by
starting paused, and reacts if the preference changes while the page is open.
Opt out with `respectReducedMotion: false` only if you have a good reason.

## Package size

The published bundle is ~21 kB minified for the CDN build. The npm package
ships **no sourcemaps** — they were roughly two thirds of the install
footprint, and the ESM/CJS builds are shipped unminified and readable. Shader
source is minified at build time, which JS minifiers cannot do because GLSL
lives in string literals.

## Browser support

Requires WebGL (WebGL 2 when available, with a WebGL 1 fallback). Without it
`initFluid` logs a warning and returns a handle whose `dispose()` still works —
it never throws, so a decorative effect cannot take down your app.

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
