import type { ISmokeyFluidConfig } from "./types";

/**
 * A named, ready-made configuration.
 *
 * Presets only set appearance and physics — never mounting or placement — so
 * they compose with whatever `canvas`, `container` or `zIndex` you pass.
 */
export type Preset = Pick<
  Partial<ISmokeyFluidConfig>,
  | "palette"
  | "colorIntensity"
  | "colorUpdateSpeed"
  | "curl"
  | "splatForce"
  | "splatRadius"
  | "densityDissipation"
  | "velocityDissipation"
  | "pressure"
  | "pressureIteration"
  | "shading"
>;

/**
 * The colour side of a preset.
 *
 * `null` means "no palette": hues are generated across the full spectrum,
 * which is the library's default behaviour.
 */
const palettes = {
  Spectrum: null,
  Sunset: ["#ff4ecd", "#ff8a4e", "#ffd24e"],
  Ocean: ["#4ea8ff", "#4effd2", "#7c4dff"],
  Mono: ["#ffffff"],
  Aurora: ["#3affa3", "#38d9ff", "#8f7bff"],
  Ember: ["#ff5722", "#ff9100", "#ffc400"],
  Lagoon: ["#00c2a8", "#00a3ff", "#0057d9"],
  Candy: ["#ff8fd0", "#ffa9f0", "#c79bff"],
  Toxic: ["#b6ff00", "#4dff88", "#00ffc8"],
  Royal: ["#5b2bff", "#8f4dff", "#c44dff"],
  Sakura: ["#ffc2dd", "#ff8fb1", "#ff6f91"],
  Mint: ["#9cffd6", "#5ef2c0", "#2fd6a5"],
  Copper: ["#ff9a5a", "#e2703a", "#b34700"],
  Ultraviolet: ["#7b2cff", "#b429ff", "#ff29f0"],
  Ice: ["#c9f0ff", "#8ad4ff", "#4fb3ff"],
  Magma: ["#ff2d2d", "#ff6a00", "#ffb300"],
  Forest: ["#2f9e44", "#69db7c", "#a9e34b"],
  Dusk: ["#3b3b98", "#7158e2", "#cd84f1"],
  Cyber: ["#00fff0", "#ff00e0", "#fffb00"],
  Pastel: ["#ffd6e0", "#c7ceea", "#b5ead7"],
} as const satisfies Record<string, readonly string[] | null>;

/**
 * The motion side of a preset — how the fluid moves, independent of colour.
 */
const characters = {
  Calm: {
    curl: 3,
    splatForce: 4200,
    splatRadius: 0.45,
    densityDissipation: 4.6,
    velocityDissipation: 2.6,
    pressureIteration: 16,
    colorUpdateSpeed: 6,
  },
  Flow: {
    curl: 10,
    splatForce: 6000,
    splatRadius: 0.5,
    densityDissipation: 3.5,
    velocityDissipation: 2,
    pressureIteration: 20,
    colorUpdateSpeed: 10,
  },
  Swirl: {
    curl: 24,
    splatForce: 7200,
    splatRadius: 0.55,
    densityDissipation: 3,
    velocityDissipation: 1.6,
    pressureIteration: 24,
    colorUpdateSpeed: 12,
  },
  Storm: {
    curl: 40,
    splatForce: 9500,
    splatRadius: 0.65,
    densityDissipation: 2.2,
    velocityDissipation: 1.2,
    pressureIteration: 28,
    colorUpdateSpeed: 16,
  },
  Wisp: {
    curl: 6,
    splatForce: 3200,
    splatRadius: 0.32,
    densityDissipation: 6.5,
    velocityDissipation: 3.4,
    pressureIteration: 12,
    colorUpdateSpeed: 8,
  },
} as const satisfies Record<string, Preset>;

export type PaletteName = keyof typeof palettes;
export type CharacterName = keyof typeof characters;
export type PresetName = `${PaletteName} ${CharacterName}`;

const build = (): Record<PresetName, Preset> => {
  const out = {} as Record<PresetName, Preset>;

  for (const [paletteName, palette] of Object.entries(palettes)) {
    for (const [characterName, character] of Object.entries(characters)) {
      const name = `${paletteName} ${characterName}` as PresetName;
      out[name] = {
        ...character,
        palette: palette ? [...palette] : null,
        // Lighter palettes need less gain to read as colour rather than glare.
        colorIntensity: palette === null ? 0.15 : 0.18,
      };
    }
  }

  return out;
};

/**
 * 100 ready-made looks: every colour palette crossed with every motion
 * character, named `"<Palette> <Character>"` — e.g. `"Ocean Swirl"`.
 *
 * ```ts
 * import { initFluid, presets } from "smokey-fluid-cursor";
 *
 * initFluid(presets["Ocean Swirl"]);
 * ```
 */
export const presets: Record<PresetName, Preset> = build();

/** Every preset name, in definition order. */
export const presetNames = Object.keys(presets) as PresetName[];

/** The palette names presets are built from. */
export const paletteNames = Object.keys(palettes) as PaletteName[];

/** The motion characters presets are built from. */
export const characterNames = Object.keys(characters) as CharacterName[];

/** Looks up a preset by name, returning `undefined` when it does not exist. */
export const getPreset = (name: string): Preset | undefined =>
  presets[name as PresetName];
