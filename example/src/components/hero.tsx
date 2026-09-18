export function Hero() {
  const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
  return (
    <header className="hero">
      <img src={`${base}/banner.svg`} alt="smokey-fluid-cursor" />
      <h1>smokey-fluid-cursor</h1>
      <p>WebGL fluid-simulation cursor trails for any website — no framework required.</p>
      <nav className="links">
        <a href="https://www.npmjs.com/package/smokey-fluid-cursor">npm</a>
        <a href="https://github.com/faraasat/smokey-fluid-cursor">GitHub</a>
        <a href="https://github.com/faraasat/smokey-fluid-cursor#readme">Docs</a>
      </nav>
    </header>
  );
}
