export function TopNav({ pkg }: { pkg: string }) {
  return (
    <div className="topnav">
      <a className="topnav__brand" href={`https://github.com/faraasat/${pkg}`}>
        <span className="topnav__dot" aria-hidden="true" />
        {pkg}
      </a>

      <nav className="topnav__links" aria-label="Project links">
        <a href={`https://www.npmjs.com/package/${pkg}`}>
          <svg width="15" height="15" viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
            <path d="M0 7.334v8h6.666v1.332H12v-1.332h12v-8H0zm6.666 6.666H5.334v-4H3.999v4H1.335V8.667h5.331v5.333zm4 0v1.336H8.001V8.667h5.334v5.333h-2.669zm12.001 0h-1.33v-4h-1.336v4h-1.335v-4h-1.33v4h-2.671V8.667h8.002v5.333zM10.665 10H12v2.667h-1.335V10z" />
          </svg>
          npm
        </a>
        <a href={`https://github.com/faraasat/${pkg}`}>
          <svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true" fill="currentColor">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
          </svg>
          GitHub
        </a>
        <a href={`https://github.com/faraasat/${pkg}#readme`}>Docs</a>
      </nav>
    </div>
  );
}
