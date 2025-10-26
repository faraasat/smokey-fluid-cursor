import {
  DoubleFBO,
  FBO,
  GL,
  GLExtInfo,
  GLFormat,
  ISmokeyFluidConfig,
  Uniforms,
} from "./types";

const defaultConfig = {
  simResolution: 128,
  dyeResolution: 1440,
  captureResolution: 512,
  densityDissipation: 3.5,
  velocityDissipation: 2,
  pressure: 0.1,
  pressureIteration: 20,
  curl: 10,
  splatRadius: 0.5,
  splatForce: 6000,
  shading: true,
  colorUpdateSpeed: 10,
  paused: false,
  backColor: { r: 0, g: 0, b: 0 },
  transparent: true,
  id: "smokey-fluid-canvas",
};

/**
 * Initializes and starts the fluid simulation
 * @param incomingConfig - Partial configuration object to override default settings
 */
export const initFluid = (incomingConfig: Partial<ISmokeyFluidConfig>) => {
  // Merge incoming config with defaults
  const config: ISmokeyFluidConfig = { ...defaultConfig, ...incomingConfig };

  // Get the canvas element by ID
  const canvas = document.getElementById(config.id) as HTMLCanvasElement | null;

  // Exit if canvas not found
  if (!canvas) return;

  // Create and inject CSS styles for the canvas
  const style = document.createElement("style");
  style.textContent = `
    #${config.id} {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: -9999;
    }
  `;
  document.head.appendChild(style);

  // Set initial canvas size
  resizeCanvas();

  /**
   * Represents a pointer (mouse/touch) for interaction
   */
  class PointerPrototype {
    id = -1; // Unique identifier for the pointer
    texcoordX = 0; // Normalized X coordinate (0-1)
    texcoordY = 0; // Normalized Y coordinate (0-1)
    prevTexcoordX = 0; // Previous normalized X coordinate
    prevTexcoordY = 0; // Previous normalized Y coordinate
    deltaX = 0; // X movement delta
    deltaY = 0; // Y movement delta
    down = false; // Whether pointer is currently pressed
    moved = false; // Whether pointer has moved since last update
    color: [number, number, number] = [30, 0, 300]; // RGB color for fluid emission
  }

  // Initialize pointers array with one pointer for mouse interaction
  const pointers: PointerPrototype[] = [];
  pointers.push(new PointerPrototype());

  // Initialize WebGL context
  const { gl, ext } = getWebGLContext(canvas);

  // Adjust configuration based on WebGL capabilities
  if (!ext.supportLinearFiltering) {
    config.dyeResolution = 512; // Lower resolution if linear filtering not supported
    config.shading = false; // Disable shading for better performance
  }

  /**
   * Gets WebGL context with appropriate extensions and capabilities
   * @param canvas - HTML canvas element
   * @returns Object containing WebGL context and extension information
   */
  function getWebGLContext(canvas: HTMLCanvasElement) {
    const params: WebGLContextAttributes = {
      alpha: true, // Enable transparency
      depth: false, // Disable depth buffer (2D simulation)
      stencil: false, // Disable stencil buffer
      antialias: false, // Disable antialiasing for performance
      preserveDrawingBuffer: false, // Don't preserve drawing buffer
    };

    // Try WebGL2 first for better performance and features
    let gl = canvas.getContext(
      "webgl2",
      params
    ) as WebGL2RenderingContext | null;

    const isWebGL2 = !!gl;

    // Fallback to WebGL1 if WebGL2 not available
    if (!isWebGL2) {
      gl =
        (canvas.getContext("webgl", params) as WebGL2RenderingContext | null) ||
        (canvas.getContext(
          "experimental-webgl",
          params
        ) as WebGL2RenderingContext | null);
    }

    // Throw error if WebGL not supported
    if (!gl) {
      throw new Error("WebGL not supported");
    }

    let halfFloat: OES_texture_half_float | null = null;
    let supportLinearFiltering: boolean;
    let halfFloatTexType: number;

    // Configure extensions based on WebGL version
    if (isWebGL2) {
      const gl2 = gl as WebGL2RenderingContext;
      // Enable color buffer float for high precision rendering
      gl2.getExtension("EXT_color_buffer_float");
      // Check if linear filtering is supported for float textures
      supportLinearFiltering = !!gl2.getExtension("OES_texture_float_linear");
      // Use native HALF_FLOAT type in WebGL2
      halfFloatTexType = gl2.HALF_FLOAT;
    } else {
      const gl1 = gl as WebGLRenderingContext;
      // Get half float extension for WebGL1
      halfFloat = gl1.getExtension(
        "OES_texture_half_float"
      ) as OES_texture_half_float | null;
      // Check linear filtering support
      supportLinearFiltering = !!gl1.getExtension(
        "OES_texture_half_float_linear"
      );
      // Throw error if half float not supported
      if (!halfFloat) {
        throw new Error("OES_texture_half_float not supported on WebGL1");
      }
      halfFloatTexType = (halfFloat as OES_texture_half_float).HALF_FLOAT_OES;
    }

    // Set clear color to black
    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    // Determine supported texture formats
    let formatRGBA: GLFormat | null = null;
    let formatRG: GLFormat | null = null;
    let formatR: GLFormat | null = null;

    if (isWebGL2) {
      const gl2 = gl as WebGL2RenderingContext;
      // Try to get supported formats for different channel counts
      formatRGBA = getSupportedFormat(
        gl2,
        gl2.RGBA16F,
        gl2.RGBA,
        halfFloatTexType
      );
      formatRG = getSupportedFormat(gl2, gl2.RG16F, gl2.RG, halfFloatTexType);
      formatR = getSupportedFormat(gl2, gl2.R16F, gl2.RED, halfFloatTexType);
    } else {
      // WebGL1: fall back to RGBA for all formats due to limited support
      const gl1 = gl as WebGLRenderingContext;
      formatRGBA = getSupportedFormat(
        gl1,
        gl1.RGBA,
        gl1.RGBA,
        halfFloatTexType
      );
      formatRG = getSupportedFormat(gl1, gl1.RGBA, gl1.RGBA, halfFloatTexType);
      formatR = getSupportedFormat(gl1, gl1.RGBA, gl1.RGBA, halfFloatTexType);
    }

    return {
      gl,
      ext: {
        formatRGBA,
        formatRG,
        formatR,
        halfFloatTexType,
        supportLinearFiltering,
        isWebGL2,
      } as GLExtInfo,
    };
  }

  /**
   * Finds the best supported texture format
   * @param gl - WebGL context
   * @param internalFormat - Preferred internal format
   * @param format - Texture format
   * @param type - Texture data type
   * @returns Supported format or null if not supported
   */
  function getSupportedFormat(
    gl: GL,
    internalFormat: number,
    format: number,
    type: number
  ): GLFormat | null {
    // Check if the preferred format is supported
    if (!supportRenderTextureFormat(gl, internalFormat, format, type)) {
      // Fallback to higher channel formats if available (WebGL2 only)
      if ((gl as WebGL2RenderingContext).RGBA16F !== undefined) {
        const gl2 = gl as WebGL2RenderingContext;
        switch (internalFormat) {
          case (gl2 as WebGL2RenderingContext).R16F:
            // Fallback from R16F to RG16F
            return getSupportedFormat(gl2, gl2.RG16F, gl2.RG, type);
          case (gl2 as WebGL2RenderingContext).RG16F:
            // Fallback from RG16F to RGBA16F
            return getSupportedFormat(gl2, gl2.RGBA16F, gl2.RGBA, type);
          default:
            return null;
        }
      }
      return null;
    }
    return { internalFormat, format };
  }

  /**
   * Tests if a specific render texture format is supported
   * @param gl - WebGL context
   * @param internalFormat - Internal texture format
   * @param format - Texture format
   * @param type - Texture data type
   * @returns Boolean indicating format support
   */
  function supportRenderTextureFormat(
    gl: GL,
    internalFormat: number,
    format: number,
    type: number
  ) {
    // Create test texture
    const texture = gl.createTexture();
    if (!texture) return false;

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Allocate texture storage
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      internalFormat,
      4,
      4,
      0,
      format,
      type,
      null
    );

    // Create framebuffer and attach texture
    const fbo = gl.createFramebuffer();
    if (!fbo) return false;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      texture,
      0
    );

    // Check framebuffer completeness
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);

    // Cleanup
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(texture);

    return status === gl.FRAMEBUFFER_COMPLETE;
  }

  /**
   * Material class for managing shaders and programs with keyword support
   */
  class Material {
    vertexShader: WebGLShader;
    fragmentShaderSource: string;
    programs: { [hash: number]: WebGLProgram };
    activeProgram: WebGLProgram | null;
    uniforms: Uniforms;

    constructor(vertexShader: WebGLShader, fragmentShaderSource: string) {
      this.vertexShader = vertexShader;
      this.fragmentShaderSource = fragmentShaderSource;
      this.programs = {};
      this.activeProgram = null;
      this.uniforms = {};
    }

    /**
     * Sets shader keywords and compiles appropriate program
     * @param keywords - Array of preprocessor keywords to enable
     */
    setKeywords(keywords: string[]) {
      // Generate hash from keywords for program caching
      let hash = 0;
      for (let i = 0; i < keywords.length; i++) hash += hashCode(keywords[i]);

      // Use cached program or create new one
      let program = this.programs[hash];

      if (!program) {
        // Compile fragment shader with keywords
        const fragmentShader = compileShader(
          gl,
          gl.FRAGMENT_SHADER,
          addKeywords(this.fragmentShaderSource, keywords)
        );
        // Create and cache program
        program = createProgram(gl, this.vertexShader, fragmentShader);
        this.programs[hash] = program;
      }

      // Skip if already using this program
      if (program === this.activeProgram) return;

      // Update uniforms and active program
      this.uniforms = getUniforms(gl, program);
      this.activeProgram = program;
    }

    /** Binds the material's active program for rendering */
    bind() {
      if (!this.activeProgram) return;
      gl.useProgram(this.activeProgram);
    }
  }

  /**
   * Simple program class for fixed shader combinations
   */
  class Program {
    program: WebGLProgram | null;
    uniforms: Uniforms;

    constructor(vertexShader: WebGLShader, fragmentShader: WebGLShader) {
      this.program = createProgram(gl, vertexShader, fragmentShader);
      this.uniforms = getUniforms(gl, this.program);
    }

    /** Binds the program for rendering */
    bind() {
      if (!this.program) return;
      gl.useProgram(this.program);
    }
  }

  /**
   * Creates a WebGL program from vertex and fragment shaders
   */
  function createProgram(
    gl: GL,
    vertexShader: WebGLShader,
    fragmentShader: WebGLShader
  ) {
    const program = gl.createProgram()!;
    // Bind attribute location 0 to "aPosition" for full-screen quad
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.bindAttribLocation(program, 0, "aPosition");
    gl.linkProgram(program);

    // Log linking errors
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.trace(gl.getProgramInfoLog(program));
    }

    return program;
  }

  /**
   * Extracts uniform locations from a shader program
   */
  function getUniforms(gl: GL, program: WebGLProgram) {
    const uniforms: Uniforms = {};
    const uniformCount: number = gl.getProgramParameter(
      program,
      gl.ACTIVE_UNIFORMS
    );

    // Get location for each uniform
    for (let i = 0; i < uniformCount; i++) {
      const info = gl.getActiveUniform(program, i);
      if (!info) continue;
      const uniformName = info.name;
      const loc = gl.getUniformLocation(program, uniformName);
      if (loc) uniforms[uniformName] = loc;
    }

    return uniforms;
  }

  /**
   * Compiles a shader from source code
   */
  function compileShader(gl: GL, type: number, source: string) {
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    // Log compilation errors
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.trace(gl.getShaderInfoLog(shader));
    }

    return shader;
  }

  /**
   * Adds preprocessor keywords to shader source
   */
  function addKeywords(source: string, keywords?: string[] | null) {
    if (!keywords || keywords.length === 0) return source;
    let keywordsString = "";
    keywords.forEach((k) => {
      keywordsString += "#define " + k + "\n";
    });
    return keywordsString + source;
  }

  // Compile base vertex shader for full-screen quad
  const baseVertexShader = compileShader(
    gl,
    gl.VERTEX_SHADER,
    `
      precision highp float;

      attribute vec2 aPosition;
      varying vec2 vUv;
      varying vec2 vL;
      varying vec2 vR;
      varying vec2 vT;
      varying vec2 vB;
      uniform vec2 texelSize;

      void main () {
          vUv = aPosition * 0.5 + 0.5;
          vL = vUv - vec2(texelSize.x, 0.0);
          vR = vUv + vec2(texelSize.x, 0.0);
          vT = vUv + vec2(0.0, texelSize.y);
          vB = vUv - vec2(0.0, texelSize.y);
          gl_Position = vec4(aPosition, 0.0, 1.0);
      }
    `
  );

  // Simple texture copy shader
  const copyShader = compileShader(
    gl,
    gl.FRAGMENT_SHADER,
    `
      precision mediump float;
      precision mediump sampler2D;

      varying highp vec2 vUv;
      uniform sampler2D uTexture;

      void main () {
          gl_FragColor = texture2D(uTexture, vUv);
      }
    `
  );

  // Shader for clearing textures with a value
  const clearShader = compileShader(
    gl,
    gl.FRAGMENT_SHADER,
    `
      precision mediump float;
      precision mediump sampler2D;

      varying highp vec2 vUv;
      uniform sampler2D uTexture;
      uniform float value;

      void main () {
          gl_FragColor = value * texture2D(uTexture, vUv);
      }
    `
  );

  // Main display shader with optional shading
  const displayShaderSource = `
      precision highp float;
      precision highp sampler2D;

      varying vec2 vUv;
      varying vec2 vL;
      varying vec2 vR;
      varying vec2 vT;
      varying vec2 vB;
      uniform sampler2D uTexture;
      uniform vec2 texelSize;

      void main () {
          vec3 c = texture2D(uTexture, vUv).rgb;

      #ifdef shading
          // Calculate normal from neighboring pixels for lighting
          vec3 lc = texture2D(uTexture, vL).rgb;
          vec3 rc = texture2D(uTexture, vR).rgb;
          vec3 tc = texture2D(uTexture, vT).rgb;
          vec3 bc = texture2D(uTexture, vB).rgb;

          float dx = length(rc) - length(lc);
          float dy = length(tc) - length(bc);

          vec3 n = normalize(vec3(dx, dy, length(texelSize)));
          vec3 l = vec3(0.0, 0.0, 1.0);

          // Apply diffuse lighting
          float diffuse = clamp(dot(n, l) + 0.7, 0.7, 1.0);
          c *= diffuse;
      #endif

          // Use brightest channel for alpha
          float a = max(c.r, max(c.g, c.b));
          gl_FragColor = vec4(c, a);
      }
    `;

  // Shader for adding splats (fluid impulses) from pointers
  const splatShader = compileShader(
    gl,
    gl.FRAGMENT_SHADER,
    `
      precision highp float;
      precision highp sampler2D;

      varying vec2 vUv;
      uniform sampler2D uTarget;
      uniform float aspectRatio;
      uniform vec3 color;
      uniform vec2 point;
      uniform float radius;

      void main () {
          // Calculate Gaussian splat
          vec2 p = vUv - point.xy;
          p.x *= aspectRatio;
          vec3 splat = exp(-dot(p, p) / radius) * color;
          vec3 base = texture2D(uTarget, vUv).xyz;
          gl_FragColor = vec4(base + splat, 1.0);
      }
    `
  );

  // Advection shader for moving fluid through velocity field
  const advectionShader = compileShader(
    gl,
    gl.FRAGMENT_SHADER,
    addKeywords(
      `
      precision highp float;
      precision highp sampler2D;

      varying vec2 vUv;
      uniform sampler2D uVelocity;
      uniform sampler2D uSource;
      uniform vec2 texelSize;
      uniform vec2 dyeTexelSize;
      uniform float dt;
      uniform float dissipation;

      // Manual bilinear filtering for WebGL1 without linear filtering support
      vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {
          vec2 st = uv / tsize - 0.5;

          vec2 iuv = floor(st);
          vec2 fuv = fract(st);

          vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
          vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
          vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
          vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);

          return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
      }

      void main () {
      #ifdef MANUAL_FILTERING
          // Manual filtering for devices without linear filtering
          vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;
          vec4 result = bilerp(uSource, coord, dyeTexelSize);
      #else
          // Standard texture lookup with linear filtering
          vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
          vec4 result = texture2D(uSource, coord);
      #endif
          // Apply dissipation (fade over time)
          float decay = 1.0 + dissipation * dt;
          gl_FragColor = result / decay;
      }
    `,
      ext.supportLinearFiltering ? null : ["MANUAL_FILTERING"]
    )
  );

  // Divergence shader for calculating velocity field divergence
  const divergenceShader = compileShader(
    gl,
    gl.FRAGMENT_SHADER,
    `
      precision mediump float;
      precision mediump sampler2D;

      varying highp vec2 vUv;
      varying highp vec2 vL;
      varying highp vec2 vR;
      varying highp vec2 vT;
      varying highp vec2 vB;
      uniform sampler2D uVelocity;

      void main () {
          // Sample neighboring velocities
          float L = texture2D(uVelocity, vL).x;
          float R = texture2D(uVelocity, vR).x;
          float T = texture2D(uVelocity, vT).y;
          float B = texture2D(uVelocity, vB).y;

          vec2 C = texture2D(uVelocity, vUv).xy;
          
          // Boundary conditions
          if (vL.x < 0.0) { L = -C.x; }
          if (vR.x > 1.0) { R = -C.x; }
          if (vT.y > 1.0) { T = -C.y; }
          if (vB.y < 0.0) { B = -C.y; }

          // Calculate divergence
          float div = 0.5 * (R - L + T - B);
          gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
      }
    `
  );

  // Curl shader for calculating vorticity (rotation) in velocity field
  const curlShader = compileShader(
    gl,
    gl.FRAGMENT_SHADER,
    `
      precision mediump float;
      precision mediump sampler2D;

      varying highp vec2 vUv;
      varying highp vec2 vL;
      varying highp vec2 vR;
      varying highp vec2 vT;
      varying highp vec2 vB;
      uniform sampler2D uVelocity;

      void main () {
          // Sample velocity components for curl calculation
          float L = texture2D(uVelocity, vL).y;
          float R = texture2D(uVelocity, vR).y;
          float T = texture2D(uVelocity, vT).x;
          float B = texture2D(uVelocity, vB).x;
          float vorticity = R - L - T + B;
          gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
      }
    `
  );

  // Vorticity shader for adding swirling motion to the fluid
  const vorticityShader = compileShader(
    gl,
    gl.FRAGMENT_SHADER,
    `
      precision highp float;
      precision highp sampler2D;

      varying vec2 vUv;
      varying vec2 vL;
      varying vec2 vR;
      varying vec2 vT;
      varying vec2 vB;
      uniform sampler2D uVelocity;
      uniform sampler2D uCurl;
      uniform float curl;
      uniform float dt;

      void main () {
          // Sample curl from neighbors
          float L = texture2D(uCurl, vL).x;
          float R = texture2D(uCurl, vR).x;
          float T = texture2D(uCurl, vT).x;
          float B = texture2D(uCurl, vB).x;
          float C = texture2D(uCurl, vUv).x;

          // Calculate vorticity force
          vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
          force /= length(force) + 0.0001; // Normalize
          force *= curl * C; // Scale by curl strength
          force.y *= -1.0; // Adjust coordinate system

          // Apply force to velocity
          vec2 velocity = texture2D(uVelocity, vUv).xy;
          velocity += force * dt;
          velocity = min(max(velocity, -1000.0), 1000.0); // Clamp to prevent explosion
          gl_FragColor = vec4(velocity, 0.0, 1.0);
      }
    `
  );

  // Pressure solver shader (Jacobi iteration)
  const pressureShader = compileShader(
    gl,
    gl.FRAGMENT_SHADER,
    `
      precision mediump float;
      precision mediump sampler2D;

      varying highp vec2 vUv;
      varying highp vec2 vL;
      varying highp vec2 vR;
      varying highp vec2 vT;
      varying highp vec2 vB;
      uniform sampler2D uPressure;
      uniform sampler2D uDivergence;

      void main () {
          // Sample pressure from neighbors
          float L = texture2D(uPressure, vL).x;
          float R = texture2D(uPressure, vR).x;
          float T = texture2D(uPressure, vT).x;
          float B = texture2D(uPressure, vB).x;
          float C = texture2D(uPressure, vUv).x;
          float divergence = texture2D(uDivergence, vUv).x;
          
          // Jacobi iteration for pressure solve
          float pressure = (L + R + B + T - divergence) * 0.25;
          gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
      }
    `
  );

  // Gradient subtract shader for making velocity field divergence-free
  const gradientSubtractShader = compileShader(
    gl,
    gl.FRAGMENT_SHADER,
    `
      precision mediump float;
      precision mediump sampler2D;

      varying highp vec2 vUv;
      varying highp vec2 vL;
      varying highp vec2 vR;
      varying highp vec2 vT;
      varying highp vec2 vB;
      uniform sampler2D uPressure;
      uniform sampler2D uVelocity;

      void main () {
          // Calculate pressure gradient
          float L = texture2D(uPressure, vL).x;
          float R = texture2D(uPressure, vR).x;
          float T = texture2D(uPressure, vT).x;
          float B = texture2D(uPressure, vB).x;
          vec2 velocity = texture2D(uVelocity, vUv).xy;
          
          // Subtract gradient to make velocity divergence-free
          velocity.xy -= vec2(R - L, T - B);
          gl_FragColor = vec4(velocity, 0.0, 1.0);
      }
    `
  );

  /**
   * Full-screen quad rendering helper (blit operation)
   * Renders a quad that covers the entire viewport
   */
  const blit = (() => {
    // Create vertex buffer for full-screen quad
    const vbo = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]),
      gl.STATIC_DRAW
    );

    // Create element buffer for triangle indices
    const ebo = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
    gl.bufferData(
      gl.ELEMENT_ARRAY_BUFFER,
      new Uint16Array([0, 1, 2, 0, 2, 3]),
      gl.STATIC_DRAW
    );

    // Set up vertex attribute
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);

    /**
     * Renders the full-screen quad to the specified target
     * @param target - Framebuffer to render to (null for screen)
     * @param clear - Whether to clear the target before rendering
     */
    return (target: FBO | null, clear = false) => {
      if (target == null) {
        // Render to screen
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      } else {
        // Render to framebuffer
        gl.viewport(0, 0, target.width, target.height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
      }

      // Clear if requested
      if (clear) {
        gl.clearColor(0.0, 0.0, 0.0, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);
      }

      // Draw the quad
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    };
  })();

  // Framebuffer objects for simulation data
  let dye: DoubleFBO; // Color/density field
  let velocity: DoubleFBO; // Velocity field
  let divergence: FBO; // Divergence field
  let curl: FBO; // Vorticity field
  let pressure: DoubleFBO; // Pressure field

  // Shader programs
  const copyProgram = new Program(baseVertexShader, copyShader);
  const clearProgram = new Program(baseVertexShader, clearShader);
  const splatProgram = new Program(baseVertexShader, splatShader);
  const advectionProgram = new Program(baseVertexShader, advectionShader);
  const divergenceProgram = new Program(baseVertexShader, divergenceShader);
  const curlProgram = new Program(baseVertexShader, curlShader);
  const vorticityProgram = new Program(baseVertexShader, vorticityShader);
  const pressureProgram = new Program(baseVertexShader, pressureShader);
  const gradienSubtractProgram = new Program(
    baseVertexShader,
    gradientSubtractShader
  );

  // Material for final display with configurable keywords
  const displayMaterial = new Material(baseVertexShader, displayShaderSource);

  /**
   * Initializes or resizes all framebuffers based on current resolution
   */
  function initFramebuffers() {
    const simRes = getResolution(config.simResolution);
    const dyeRes = getResolution(config.dyeResolution);

    const texType = ext.halfFloatTexType;
    const rgba = ext.formatRGBA!;
    const rg = ext.formatRG!;
    const r = ext.formatR!;
    const filtering = ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST;

    gl.disable(gl.BLEND);

    // Initialize or resize dye framebuffers
    if (dye == null)
      dye = createDoubleFBO(
        dyeRes.width,
        dyeRes.height,
        rgba.internalFormat,
        rgba.format,
        texType,
        filtering
      );
    else
      dye = resizeDoubleFBO(
        dye,
        dyeRes.width,
        dyeRes.height,
        rgba.internalFormat,
        rgba.format,
        texType,
        filtering
      );

    // Initialize or resize velocity framebuffers
    if (velocity == null)
      velocity = createDoubleFBO(
        simRes.width,
        simRes.height,
        rg.internalFormat,
        rg.format,
        texType,
        filtering
      );
    else
      velocity = resizeDoubleFBO(
        velocity,
        simRes.width,
        simRes.height,
        rg.internalFormat,
        rg.format,
        texType,
        filtering
      );

    // Create single FBOs for intermediate calculations
    divergence = createFBO(
      simRes.width,
      simRes.height,
      r.internalFormat,
      r.format,
      texType,
      gl.NEAREST
    );
    curl = createFBO(
      simRes.width,
      simRes.height,
      r.internalFormat,
      r.format,
      texType,
      gl.NEAREST
    );
    pressure = createDoubleFBO(
      simRes.width,
      simRes.height,
      r.internalFormat,
      r.format,
      texType,
      gl.NEAREST
    );
  }

  /**
   * Creates a single framebuffer object
   */
  function createFBO(
    w: number,
    h: number,
    internalFormat: number,
    format: number,
    type: number,
    param: number
  ): FBO {
    gl.activeTexture(gl.TEXTURE0);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      internalFormat,
      w,
      h,
      0,
      format,
      type,
      null
    );

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      texture,
      0
    );
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const texelSizeX = 1.0 / w;
    const texelSizeY = 1.0 / h;

    return {
      texture,
      fbo,
      width: w,
      height: h,
      texelSizeX,
      texelSizeY,
      attach(id: number) {
        gl.activeTexture(gl.TEXTURE0 + id);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        return id;
      },
    };
  }

  /**
   * Creates a double framebuffer for ping-pong rendering
   */
  function createDoubleFBO(
    w: number,
    h: number,
    internalFormat: number,
    format: number,
    type: number,
    param: number
  ): DoubleFBO {
    let fbo1 = createFBO(w, h, internalFormat, format, type, param);
    let fbo2 = createFBO(w, h, internalFormat, format, type, param);

    return {
      width: w,
      height: h,
      texelSizeX: fbo1.texelSizeX,
      texelSizeY: fbo1.texelSizeY,
      get read() {
        return fbo1;
      },
      set read(value: FBO) {
        fbo1 = value;
      },
      get write() {
        return fbo2;
      },
      set write(value: FBO) {
        fbo2 = value;
      },
      swap() {
        const temp = fbo1;
        fbo1 = fbo2;
        fbo2 = temp;
      },
    };
  }

  /**
   * Resizes a framebuffer and copies its contents
   */
  function resizeFBO(
    target: FBO,
    w: number,
    h: number,
    internalFormat: number,
    format: number,
    type: number,
    param: number
  ): FBO {
    const newFBO = createFBO(w, h, internalFormat, format, type, param);
    copyProgram.bind();
    gl.uniform1i(copyProgram.uniforms["uTexture"], target.attach(0));
    blit(newFBO);
    return newFBO;
  }

  /**
   * Resizes a double framebuffer
   */
  function resizeDoubleFBO(
    target: DoubleFBO,
    w: number,
    h: number,
    internalFormat: number,
    format: number,
    type: number,
    param: number
  ): DoubleFBO {
    if (target.width === w && target.height === h) return target;
    target.read = resizeFBO(
      target.read,
      w,
      h,
      internalFormat,
      format,
      type,
      param
    );
    target.write = createFBO(w, h, internalFormat, format, type, param);
    target.width = w;
    target.height = h;
    target.texelSizeX = 1.0 / w;
    target.texelSizeY = 1.0 / h;
    return target;
  }

  /**
   * Updates shader keywords based on configuration
   */
  function updateKeywords() {
    const displayKeywords: string[] = [];
    if (config.shading) displayKeywords.push("shading");
    displayMaterial.setKeywords(displayKeywords);
  }

  // Initialize simulation
  updateKeywords();
  initFramebuffers();

  // Animation state
  let lastUpdateTime = Date.now();
  let colorUpdateTimer = 0.0;

  /**
   * Main animation loop
   */
  function update() {
    const dt = calcDeltaTime();
    if (resizeCanvas()) initFramebuffers();
    updateColors(dt);
    applyInputs();
    step(dt);
    render(null);
    requestAnimationFrame(update);
  }

  /**
   * Calculates time delta since last frame
   */
  function calcDeltaTime() {
    const now = Date.now();
    let dt = (now - lastUpdateTime) / 1000;
    dt = Math.min(dt, 0.016666); // Cap at 60 FPS
    lastUpdateTime = now;
    return dt;
  }

  /**
   * Resizes canvas to match display size
   * @returns Boolean indicating if resize occurred
   */
  function resizeCanvas() {
    const width = scaleByPixelRatio(canvas!.clientWidth);
    const height = scaleByPixelRatio(canvas!.clientHeight);
    if (canvas!.width !== width || canvas!.height !== height) {
      canvas!.width = width;
      canvas!.height = height;
      return true;
    }
    return false;
  }

  /**
   * Updates pointer colors over time
   */
  function updateColors(dt: number) {
    colorUpdateTimer += dt * config.colorUpdateSpeed;
    if (colorUpdateTimer >= 1) {
      colorUpdateTimer = wrap(colorUpdateTimer, 0, 1);
      pointers.forEach((p) => {
        p.color = rgbToTuple(generateColor());
      });
    }
  }

  /**
   * Applies pointer inputs to the simulation
   */
  function applyInputs() {
    pointers.forEach((p) => {
      if (p.moved) {
        p.moved = false;
        splatPointer(p);
      }
    });
  }

  /**
   * Performs one simulation step
   */
  function step(dt: number) {
    gl.disable(gl.BLEND);

    // Curl calculation
    curlProgram.bind();
    gl.uniform2f(
      curlProgram.uniforms["texelSize"],
      velocity.texelSizeX,
      velocity.texelSizeY
    );
    gl.uniform1i(curlProgram.uniforms["uVelocity"], velocity.read.attach(0));
    blit(curl);

    // Vorticity confinement
    vorticityProgram.bind();
    gl.uniform2f(
      vorticityProgram.uniforms["texelSize"],
      velocity.texelSizeX,
      velocity.texelSizeY
    );
    gl.uniform1i(
      vorticityProgram.uniforms["uVelocity"],
      velocity.read.attach(0)
    );
    gl.uniform1i(vorticityProgram.uniforms["uCurl"], curl.attach(1));
    gl.uniform1f(vorticityProgram.uniforms["curl"], config.curl);
    gl.uniform1f(vorticityProgram.uniforms["dt"], dt);
    blit(velocity.write);
    velocity.swap();

    // Divergence calculation
    divergenceProgram.bind();
    gl.uniform2f(
      divergenceProgram.uniforms["texelSize"],
      velocity.texelSizeX,
      velocity.texelSizeY
    );
    gl.uniform1i(
      divergenceProgram.uniforms["uVelocity"],
      velocity.read.attach(0)
    );
    blit(divergence);

    // Clear pressure
    clearProgram.bind();
    gl.uniform1i(clearProgram.uniforms["uTexture"], pressure.read.attach(0));
    gl.uniform1f(clearProgram.uniforms["value"], config.pressure);
    blit(pressure.write);
    pressure.swap();

    // Pressure solve (Jacobi iterations)
    pressureProgram.bind();
    gl.uniform2f(
      pressureProgram.uniforms["texelSize"],
      velocity.texelSizeX,
      velocity.texelSizeY
    );
    gl.uniform1i(pressureProgram.uniforms["uDivergence"], divergence.attach(0));
    for (let i = 0; i < config.pressureIteration; i++) {
      gl.uniform1i(
        pressureProgram.uniforms["uPressure"],
        pressure.read.attach(1)
      );
      blit(pressure.write);
      pressure.swap();
    }

    // Gradient subtraction to make velocity divergence-free
    gradienSubtractProgram.bind();
    gl.uniform2f(
      gradienSubtractProgram.uniforms["texelSize"],
      velocity.texelSizeX,
      velocity.texelSizeY
    );
    gl.uniform1i(
      gradienSubtractProgram.uniforms["uPressure"],
      pressure.read.attach(0)
    );
    gl.uniform1i(
      gradienSubtractProgram.uniforms["uVelocity"],
      velocity.read.attach(1)
    );
    blit(velocity.write);
    velocity.swap();

    // Advect velocity
    advectionProgram.bind();
    gl.uniform2f(
      advectionProgram.uniforms["texelSize"],
      velocity.texelSizeX,
      velocity.texelSizeY
    );
    if (!ext.supportLinearFiltering) {
      gl.uniform2f(
        advectionProgram.uniforms["dyeTexelSize"],
        velocity.texelSizeX,
        velocity.texelSizeY
      );
    }
    const velocityId = velocity.read.attach(0);
    gl.uniform1i(advectionProgram.uniforms["uVelocity"], velocityId);
    gl.uniform1i(advectionProgram.uniforms["uSource"], velocityId);
    gl.uniform1f(advectionProgram.uniforms["dt"], dt);
    gl.uniform1f(
      advectionProgram.uniforms["dissipation"],
      config.velocityDissipation
    );
    blit(velocity.write);
    velocity.swap();

    // Advect dye
    if (!ext.supportLinearFiltering) {
      gl.uniform2f(
        advectionProgram.uniforms["dyeTexelSize"],
        dye.texelSizeX,
        dye.texelSizeY
      );
    }
    gl.uniform1i(
      advectionProgram.uniforms["uVelocity"],
      velocity.read.attach(0)
    );
    gl.uniform1i(advectionProgram.uniforms["uSource"], dye.read.attach(1));
    gl.uniform1f(
      advectionProgram.uniforms["dissipation"],
      config.densityDissipation
    );
    blit(dye.write);
    dye.swap();
  }

  /**
   * Renders the final result
   */
  function render(target: FBO | null) {
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.enable(gl.BLEND);
    drawDisplay(target);
  }

  /**
   * Draws the final display with optional shading
   */
  function drawDisplay(target: FBO | null) {
    const width = target == null ? gl.drawingBufferWidth : target.width;
    const height = target == null ? gl.drawingBufferHeight : target.height;

    displayMaterial.bind();
    if (config.shading) {
      gl.uniform2f(
        displayMaterial.uniforms["texelSize"],
        1.0 / width,
        1.0 / height
      );
    }
    gl.uniform1i(displayMaterial.uniforms["uTexture"], dye.read.attach(0));
    blit(target);
  }

  /**
   * Applies a splat from pointer movement
   */
  function splatPointer(pointer: PointerPrototype) {
    const dx = pointer.deltaX * config.splatForce;
    const dy = pointer.deltaY * config.splatForce;
    const color = tupleToRgb(pointer.color);
    splat(pointer.texcoordX, pointer.texcoordY, dx, dy, color);
  }

  /**
   * Applies a click splat (stronger impulse)
   */
  function clickSplat(pointer: PointerPrototype) {
    const color = generateColor();
    color.r *= 10.0;
    color.g *= 10.0;
    color.b *= 10.0;
    const dx = 10 * (Math.random() - 0.5);
    const dy = 30 * (Math.random() - 0.5);
    splat(pointer.texcoordX, pointer.texcoordY, dx, dy, color);
  }

  /**
   * Applies a fluid splat at specified coordinates
   */
  function splat(
    x: number,
    y: number,
    dx: number,
    dy: number,
    color: { r: number; g: number; b: number }
  ) {
    // Splat velocity
    splatProgram.bind();
    gl.uniform1i(splatProgram.uniforms["uTarget"], velocity.read.attach(0));
    gl.uniform1f(
      splatProgram.uniforms["aspectRatio"],
      canvas!.width / canvas!.height
    );
    gl.uniform2f(splatProgram.uniforms["point"], x, y);
    gl.uniform3f(splatProgram.uniforms["color"], dx, dy, 0.0);
    gl.uniform1f(
      splatProgram.uniforms["radius"],
      correctRadius(config.splatRadius / 100.0)
    );
    blit(velocity.write);
    velocity.swap();

    // Splat dye
    gl.uniform1i(splatProgram.uniforms["uTarget"], dye.read.attach(0));
    gl.uniform3f(splatProgram.uniforms["color"], color.r, color.g, color.b);
    blit(dye.write);
    dye.swap();
  }

  /**
   * Adjusts splat radius based on aspect ratio
   */
  function correctRadius(radius: number) {
    const aspectRatio = canvas!.width / canvas!.height;
    if (aspectRatio > 1) radius *= aspectRatio;
    return radius;
  }

  // Event listeners for user interaction

  window.addEventListener("mousedown", (e) => {
    const pointer = pointers[0];
    const rect = canvas.getBoundingClientRect();
    const posX = scaleByPixelRatio(e.clientX - rect.left);
    const posY = scaleByPixelRatio(e.clientY - rect.top);
    updatePointerDownData(pointer, -1, posX, posY);
    clickSplat(pointer);
  });

  window.addEventListener("mousemove", (e) => {
    const pointer = pointers[0];
    const rect = canvas.getBoundingClientRect();
    const posX = scaleByPixelRatio(e.clientX - rect.left);
    const posY = scaleByPixelRatio(e.clientY - rect.top);
    const color = pointer.color;
    updatePointerMoveData(pointer, posX, posY, color);
  });

  window.addEventListener("touchstart", (e) => {
    const touches = e.targetTouches;
    const rect = canvas.getBoundingClientRect();
    const pointer = pointers[0];
    for (let i = 0; i < touches.length; i++) {
      const posX = scaleByPixelRatio(touches[i].clientX - rect.left);
      const posY = scaleByPixelRatio(touches[i].clientY - rect.top);
      updatePointerDownData(pointer, touches[i].identifier, posX, posY);
      clickSplat(pointer);
    }
  });

  window.addEventListener(
    "touchmove",
    (e) => {
      e.preventDefault();
      const touches = e.targetTouches;
      const rect = canvas.getBoundingClientRect();
      const pointer = pointers[0];
      for (let i = 0; i < touches.length; i++) {
        const posX = scaleByPixelRatio(touches[i].clientX - rect.left);
        const posY = scaleByPixelRatio(touches[i].clientY - rect.top);
        updatePointerMoveData(pointer, posX, posY, pointer.color);
      }
    },
    { passive: false }
  );

  window.addEventListener("touchend", (e) => {
    const touches = e.changedTouches;
    const pointer = pointers[0];
    for (let i = 0; i < touches.length; i++) {
      updatePointerUpData(pointer);
    }
  });

  /**
   * Updates pointer data when pressed down
   */
  function updatePointerDownData(
    pointer: PointerPrototype,
    id: number,
    posX: number,
    posY: number
  ) {
    pointer.id = id;
    pointer.down = true;
    pointer.moved = false;
    pointer.texcoordX = posX / canvas!.width;
    pointer.texcoordY = 1.0 - posY / canvas!.height;
    pointer.prevTexcoordX = pointer.texcoordX;
    pointer.prevTexcoordY = pointer.texcoordY;
    pointer.deltaX = 0;
    pointer.deltaY = 0;
    pointer.color = rgbToTuple(generateColor());
  }

  /**
   * Updates pointer data during movement
   */
  function updatePointerMoveData(
    pointer: PointerPrototype,
    posX: number,
    posY: number,
    color: [number, number, number]
  ) {
    pointer.prevTexcoordX = pointer.texcoordX;
    pointer.prevTexcoordY = pointer.texcoordY;
    pointer.texcoordX = posX / canvas!.width;
    pointer.texcoordY = 1.0 - posY / canvas!.height;
    pointer.deltaX = correctDeltaX(pointer.texcoordX - pointer.prevTexcoordX);
    pointer.deltaY = correctDeltaY(pointer.texcoordY - pointer.prevTexcoordY);
    pointer.moved =
      Math.abs(pointer.deltaX) > 0 || Math.abs(pointer.deltaY) > 0;
    pointer.color = color;
  }

  /**
   * Updates pointer data when released
   */
  function updatePointerUpData(pointer: PointerPrototype) {
    pointer.down = false;
  }

  /**
   * Corrects X delta based on aspect ratio
   */
  function correctDeltaX(delta: number) {
    const aspectRatio = canvas!.width / canvas!.height;
    if (aspectRatio < 1) delta *= aspectRatio;
    return delta;
  }

  /**
   * Corrects Y delta based on aspect ratio
   */
  function correctDeltaY(delta: number) {
    const aspectRatio = canvas!.width / canvas!.height;
    if (aspectRatio > 1) delta /= aspectRatio;
    return delta;
  }

  /**
   * Generates a random color with reduced intensity
   */
  function generateColor() {
    const c = HSVtoRGB(Math.random(), 1.0, 1.0);
    c.r *= 0.15;
    c.g *= 0.15;
    c.b *= 0.15;
    return c;
  }

  /**
   * Converts RGB object to tuple
   */
  function rgbToTuple(c: {
    r: number;
    g: number;
    b: number;
  }): [number, number, number] {
    return [c.r, c.g, c.b];
  }

  /**
   * Converts tuple to RGB object
   */
  function tupleToRgb(t: [number, number, number]) {
    return { r: t[0], g: t[1], b: t[2] };
  }

  /**
   * Converts HSV color to RGB
   */
  function HSVtoRGB(h: number, s: number, v: number) {
    let r = 0,
      g = 0,
      b = 0;
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);

    switch (i % 6) {
      case 0:
        r = v;
        g = t;
        b = p;
        break;
      case 1:
        r = q;
        g = v;
        b = p;
        break;
      case 2:
        r = p;
        g = v;
        b = t;
        break;
      case 3:
        r = p;
        g = q;
        b = v;
        break;
      case 4:
        r = t;
        g = p;
        b = v;
        break;
      case 5:
        r = v;
        g = p;
        b = q;
        break;
    }

    return { r, g, b };
  }

  /**
   * Wraps a value between min and max
   */
  function wrap(value: number, min: number, max: number) {
    const range = max - min;
    if (range === 0) return min;
    return ((value - min) % range) + min;
  }

  /**
   * Calculates resolution based on aspect ratio
   */
  function getResolution(resolution: number) {
    let aspectRatio = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (aspectRatio < 1) aspectRatio = 1.0 / aspectRatio;

    const min = Math.round(resolution);
    const max = Math.round(resolution * aspectRatio);

    if (gl.drawingBufferWidth > gl.drawingBufferHeight)
      return { width: max, height: min };
    else return { width: min, height: max };
  }

  /**
   * Scales value by device pixel ratio for crisp rendering
   */
  function scaleByPixelRatio(input: number) {
    const pixelRatio = window.devicePixelRatio || 1;
    return Math.floor(input * pixelRatio);
  }

  /**
   * Generates hash code for string (used for shader caching)
   */
  function hashCode(s: string) {
    if (s.length === 0) return 0;
    let hash = 0;
    for (let i = 0; i < s.length; i++) {
      hash = (hash << 5) - hash + s.charCodeAt(i);
      hash |= 0;
    }
    return hash;
  }

  // Start the simulation loop
  update();
};
