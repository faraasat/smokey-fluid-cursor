<h1 align="center">🌊 Smokey Fluid Cursor</h1>

<p align="center">
A beautiful, interactive fluid simulation that creates stunning visual effects following your cursor movements. Built with WebGL for high-performance real-time fluid dynamics.
</p>

![npm version](https://img.shields.io/npm/v/smokey-fluid-cursor.svg)
![package size minified](https://img.shields.io/bundlephobia/min/smokey-fluid-cursor?style=plastic)
[![Badge](https://data.jsdelivr.com/v1/package/npm/smokey-fluid-cursor/badge)](https://www.jsdelivr.com/package/npm/smokey-fluid-cursor)
[![JavaScript Style Guide](https://img.shields.io/badge/code_style-standard-brightgreen.svg)](https://standardjs.com)

![total downloads](https://img.shields.io/npm/dt/smokey-fluid-cursor.svg)
![total downloads per year](https://img.shields.io/npm/dy/smokey-fluid-cursor.svg)
![total downloads per week](https://img.shields.io/npm/dw/smokey-fluid-cursor.svg)
![total downloads per month](https://img.shields.io/npm/dm/smokey-fluid-cursor.svg)
![download-image](https://img.shields.io/npm/dm/smokey-fluid-cursor.svg)


[![smokey-fluid-cursor](https://nodei.co/npm/smokey-fluid-cursor.png)](https://npmjs.org/package/smokey-fluid-cursor)

---

## 📦 Installation

```bash
npm i smokey-fluid-cursor

yarn add smokey-fluid-cursor

pnpm i smokey-fluid-cursor

bun add smokey-fluid-cursor
```

---

## 🚀 Quick Start

### Basic Usage

```javascript
import { initFluid } from "smokey-fluid-cursor";

// Initialize with default settings
initFluid();
```

### HTML Setup

```html
<!DOCTYPE html>
<html>
  <head>
    <style>
      body {
        margin: 0;
        background: #000;
        overflow: hidden;
      }
    </style>
  </head>
  <body>
    <canvas id="smokey-fluid-canvas"></canvas>

    <script type="module">
      import { initFluid } from "smokey-fluid-cursor/index.mjs";

      window.addEventListener("DOMContentLoaded", () => {
        initFluid();
      });
    </script>
  </body>
</html>
```

### IIFE Setup

```html
<!DOCTYPE html>
<html>
  <head>
    <style>
      body {
        margin: 0;
        background: #000;
        overflow: hidden;
      }
    </style>
  </head>
  <body>
    <canvas id="smokey-fluid-canvas"></canvas>

    <script src="smokey-fluid-cursor/index.global.js"></script>

    <script type="module">
      window.addEventListener("DOMContentLoaded", () => {
        if (window.SmokyFluid) {
          const fluid = SmokyFluid.initFluid();
        } else {
          console.error("SmokyFluid not found!");
        }
      });
    </script>
  </body>
</html>
```

---

## ⚙️ Configuration

Customize the fluid simulation with these configuration options:

```javascript
initFluid({
  id: "smokey-fluid-cursor", // Canvas element ID
  simResolution: 128, // Simulation resolution (higher = more detailed)
  dyeResolution: 512, // Dye/color resolution
  densityDissipation: 0.98, // How quickly colors fade (0-1)
  velocityDissipation: 0.98, // How quickly movement slows down
  pressureIteration: 10, // Pressure solver iterations
  curl: 30, // Vorticity/swirl intensity
  splatRadius: 0.25, // Size of cursor splats
  splatForce: 6000, // Force of cursor movements
  shading: true, // Enable 3D lighting effects
  colorUpdateSpeed: 0.5, // Speed of color transitions
  transparent: false, // Transparent background
});
```

---

## 🌟 Features

- **Real-time Fluid Dynamics**: Physics-based simulation using Navier-Stokes equations
- **WebGL Accelerated**: High-performance rendering for smooth 60fps
- **Interactive**: Responds to mouse and touch movements
- **Customizable**: Extensive configuration options
- **Mobile Support**: Touch-optimized interactions
- **Auto-scaling**: Adapts to screen size and pixel ratio
- **Color Cycling**: Dynamic, evolving color palettes
- **3D Lighting**: Optional shading for depth perception

---

## 🎯 Use Cases

- **Website Backgrounds**: Immersive animated backgrounds
- **Cursor Effects**: Enhanced user interaction feedback
- **Data Visualization**: Fluid-based data representations
- **Art Installations**: Digital art and creative coding
- **Game Effects**: Atmospheric and UI effects
- **Product Demos**: Eye-catching technology showcases

---

## 🧑‍💻 Author

Built and maintained by [**Farasat Ali**](https://www.farasat.me)

- Website: [www.farasat.me](https://www.farasat.me)
- LinkedIn: [linkedin.com/in/faraasat](https://linkedin.com/in/faraasat)
- GitHub: [github.com/faraasat](https://github.com/faraasat)
