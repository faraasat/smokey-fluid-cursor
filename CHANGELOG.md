# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

## [2.1.0](https://github.com/faraasat/smokey-fluid-cursor/compare/v2.0.0...v2.1.0) (2026-09-30)


### Features

* 100 presets, syntax-highlighted demo, top nav ([7759319](https://github.com/faraasat/smokey-fluid-cursor/commit/7759319e71666a80548b0ada76e8db86a38ebabe))


### Bug Fixes

* the cursor effect was painted over and invisible ([a145730](https://github.com/faraasat/smokey-fluid-cursor/commit/a145730415b4aca90dde8aa3a2aa9c802fff0a25))

## [2.0.0](https://github.com/faraasat/smokey-fluid-cursor/compare/v1.0.7...v2.0.0) (2026-09-28)


### Features

* mountable/positionable canvas, live controls, and a build-time GLSL minifier that keeps the JavaScript payload flat at ~21 kB despite the new APIs (note: the original commit subject claimed "58% smaller package", which compared against an intermediate branch state rather than the published 1.0.7 — see the size note below) ([a769c1d](https://github.com/faraasat/smokey-fluid-cursor/commit/a769c1d0aea6fc3d9e26ad210af12756dfdbaa72))


### Package size

Measured against the published 1.0.7, this release is **larger**, not smaller:
22 kB packed vs 9 kB. The JavaScript payload is essentially unchanged; the
growth is the `index.d.mts` required for correct ESM type resolution, much
richer JSDoc in the declarations, and genuinely more code.


### Bug Fixes

* drop the broken ./iife subpath export, plus a11y audit ([6908fd9](https://github.com/faraasat/smokey-fluid-cursor/commit/6908fd960a73e4cb99b3551f54e8bd76b5a01370))

### 1.0.7 (2025-10-26)

### 1.0.6 (2025-10-26)


### Bug Fixes

* readme ([876cdbe](https://github.com/faraasat/smokey-fluid-cursor/commit/876cdbeadcecff18794aa02ca127d8c3f8dfb489))

### 1.0.5 (2025-10-26)

### 1.0.4 (2025-10-26)

### 1.0.3 (2025-10-26)

### 1.0.2 (2025-10-26)

### 1.0.1 (2025-10-26)
