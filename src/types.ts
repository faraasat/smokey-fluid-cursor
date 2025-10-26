/**
 * This file defines the foundational TypeScript interfaces
 * and type aliases used throughout the Smokey Fluid simulation engine.
 *
 * It abstracts WebGL object types (FBOs, textures, uniforms, etc.)
 * to provide clear, type-safe contracts for interacting with GPU resources.
 *
 * Developers integrating or extending this package should primarily
 * adjust configuration options defined in `ISmokeyFluidConfig`.
 */

/**
 * Generic alias for either WebGL1 or WebGL2 rendering contexts.
 *
 * - WebGLRenderingContext → legacy WebGL1 API
 * - WebGL2RenderingContext → modern API with extended texture formats and MRT support
 *
 * This alias allows code to remain compatible across both WebGL versions.
 */
export type GL = WebGLRenderingContext | WebGL2RenderingContext;

/**
 * Represents a simple mapping of uniform variable names to their
 * corresponding compiled WebGL uniform locations.
 *
 * Example:
 * ```ts
 * const uniforms: Uniforms = {
 *   uTime: gl.getUniformLocation(program, "uTime"),
 *   uResolution: gl.getUniformLocation(program, "uResolution"),
 * };
 * ```
 */
export type Uniforms = Record<string, WebGLUniformLocation>;

/**
 * Describes a format pair used for texture or renderbuffer configuration.
 *
 * - `internalFormat`: how GPU stores the texture internally (e.g., RGBA16F)
 * - `format`: how the data is laid out when uploading pixels (e.g., RGBA)
 *
 * Used for compatibility mapping between WebGL1 and WebGL2.
 */
export interface GLFormat {
  internalFormat: number;
  format: number;
}

/**
 * Represents detected WebGL extension and format capabilities
 * for the current context. This allows the engine to dynamically
 * choose the optimal texture formats and filtering modes.
 *
 * Usually populated at initialization after checking for extensions like:
 * - OES_texture_half_float
 * - OES_texture_float_linear
 * - EXT_color_buffer_float
 */
export interface GLExtInfo {
  /** Supported RGBA floating-point texture format */
  formatRGBA: GLFormat | null;

  /** Supported RG (two-channel) texture format (WebGL2 only) */
  formatRG: GLFormat | null;

  /** Supported R (single-channel) texture format (WebGL2 only) */
  formatR: GLFormat | null;

  /** GL constant for HALF_FLOAT type (e.g., gl.HALF_FLOAT_OES) */
  halfFloatTexType: number;

  /** Whether linear filtering (smooth interpolation) for float textures is available */
  supportLinearFiltering: boolean;

  /** True if running on WebGL2 (vs WebGL1 + extensions) */
  isWebGL2: boolean;
}

/**
 * Framebuffer Object (FBO) descriptor — wraps a texture and its framebuffer target.
 *
 * Each FBO represents a GPU render target that can store color or data fields
 * (such as velocity, pressure, or density in fluid simulation).
 */
export interface FBO {
  /** GPU texture attached to the framebuffer */
  texture: WebGLTexture | null;

  /** The framebuffer object itself */
  fbo: WebGLFramebuffer | null;

  /** Width of the texture in pixels */
  width: number;

  /** Height of the texture in pixels */
  height: number;

  /** Normalized texel size (1 / width) */
  texelSizeX: number;

  /** Normalized texel size (1 / height) */
  texelSizeY: number;

  /**
   * Binds this FBO's texture to a given texture unit (slot).
   * Returns the bound texture unit index for convenience.
   *
   * Example:
   * ```ts
   * const unit = velocityFBO.attach(0); // binds to TEXTURE0
   * gl.uniform1i(uniforms.uVelocity, unit);
   * ```
   */
  attach(id: number): number;
}

/**
 * Represents a double-buffered framebuffer system.
 *
 * Used heavily in fluid simulations to "ping-pong" between two framebuffers:
 * - One for reading previous state
 * - One for writing the next frame
 *
 * After each simulation step, `swap()` is called to exchange them.
 */
export interface DoubleFBO {
  /** Buffer width in pixels */
  width: number;

  /** Buffer height in pixels */
  height: number;

  /** Normalized texel size (1 / width) */
  texelSizeX: number;

  /** Normalized texel size (1 / height) */
  texelSizeY: number;

  /** Framebuffer currently used for reading */
  read: FBO;

  /** Framebuffer currently used for writing */
  write: FBO;

  /** Swaps read/write buffers for next iteration */
  swap(): void;
}

/**
 * ---------------------------------------------------------
 * 🔧 Smokey Fluid Configuration
 * ---------------------------------------------------------
 *
 * This interface defines **all configurable simulation parameters**
 * that can be customized externally (e.g., by UI, app settings, or user code).
 *
 * Each property controls a different aspect of the fluid’s physics,
 * rendering behavior, or visual appearance.
 */
export interface ISmokeyFluidConfig {
  /** Simulation grid resolution for velocity/pressure fields (lower = faster but coarser) */
  simResolution: number;

  /** Visual color/dye buffer resolution (affects rendering detail) */
  dyeResolution: number;

  /** Resolution used for high-quality screenshots or recording */
  captureResolution: number;

  /** Rate at which color fades from the fluid (higher = faster fade) */
  densityDissipation: number;

  /** Rate at which velocity energy dissipates (higher = faster slowdown) */
  velocityDissipation: number;

  /** Initial pressure clear multiplier (affects stability of solver) */
  pressure: number;

  /** Number of Jacobi iterations when solving pressure (higher = more accurate but slower) */
  pressureIteration: number;

  /** Strength of vorticity confinement (adds swirling, curl-like motion) */
  curl: number;

  /** Base normalized radius of user splats (0–1 range relative to screen size) */
  splatRadius: number;

  /** Force multiplier applied when user interacts (e.g., mouse/touch input) */
  splatForce: number;

  /** Enable or disable lighting/shading effects for visual depth */
  shading: boolean;

  /** Speed at which the dynamic color palette rotates during simulation */
  colorUpdateSpeed: number;

  /** If true, pauses the main simulation loop (used for debugging or static rendering) */
  paused: boolean;

  /**
   * Canvas background color (can be in 0–1 normalized range or 0–255)
   * Example: `{ r: 0, g: 0, b: 0 }` for black.
   */
  backColor: { r: number; g: number; b: number };

  /**
   * Determines whether the canvas should preserve alpha transparency.
   * If `true`, blending with HTML backgrounds is allowed.
   */
  transparent: boolean;

  /**
   * ID assigned to the canvas element.
   * Default is "smokey-fluid-canvas".
   */
  id: string;
}
