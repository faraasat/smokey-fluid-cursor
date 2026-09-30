"use client";

import { useState } from "react";
import { Highlight, type PrismTheme } from "prism-react-renderer";

/**
 * Theme derived from the demo's own CSS custom properties, so the highlighting
 * matches the surrounding page rather than fighting it.
 */
const theme: PrismTheme = {
  plain: { color: "#e8eef8", backgroundColor: "transparent" },
  styles: [
    { types: ["comment", "prolog", "cdata"], style: { color: "#7b8da8", fontStyle: "italic" } },
    { types: ["punctuation"], style: { color: "#8fa3c0" } },
    { types: ["tag", "operator", "keyword", "selector"], style: { color: "#ff8bd1" } },
    { types: ["function", "class-name", "function-variable"], style: { color: "#78d2ff" } },
    { types: ["string", "char", "attr-value", "template-string"], style: { color: "#9fe8a8" } },
    { types: ["number", "boolean", "constant", "symbol"], style: { color: "#ffc978" } },
    { types: ["attr-name", "property", "variable"], style: { color: "#c8b4ff" } },
    { types: ["builtin", "regex"], style: { color: "#7fe3d4" } },
    { types: ["deleted"], style: { color: "#ff92a5" } },
    { types: ["inserted"], style: { color: "#6ee7a8" } },
  ],
};

type Language = "tsx" | "ts" | "jsx" | "js" | "bash" | "css" | "json" | "html";

export function Code({
  children,
  language = "tsx",
  label,
}: {
  children: string;
  language?: Language;
  label?: string;
}) {
  const [copied, setCopied] = useState(false);
  const code = children.trim();

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard blocked — the code is still selectable */
    }
  };

  return (
    <figure className="code">
      <figcaption className="code__bar">
        <span className="code__lang">{label ?? language}</span>
        <button
          type="button"
          className="code__copy"
          onClick={copy}
          aria-label={copied ? "Copied" : "Copy code to clipboard"}
        >
          {copied ? (
            <>
              <svg width="13" height="13" viewBox="0 0 16 16" aria-hidden="true">
                <path d="M2 8.5 6 12.5 14 4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Copied
            </>
          ) : (
            <>
              <svg width="13" height="13" viewBox="0 0 16 16" aria-hidden="true">
                <rect x="5.5" y="5.5" width="9" height="9" rx="1.6" fill="none" stroke="currentColor" strokeWidth="1.6" />
                <path d="M10.5 5.5v-2a1.5 1.5 0 0 0-1.5-1.5H3a1.5 1.5 0 0 0-1.5 1.5V9A1.5 1.5 0 0 0 3 10.5h2" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              </svg>
              Copy
            </>
          )}
        </button>
      </figcaption>

      <Highlight theme={theme} code={code} language={language}>
        {({ style, tokens, getLineProps, getTokenProps }) => (
          // tabIndex keeps the horizontally scrollable region reachable by
          // keyboard (axe: scrollable-region-focusable).
          <pre className="code__pre" style={style} tabIndex={0}>
            {tokens.map((line, i) => (
              <span key={i} {...getLineProps({ line })} className="code__line">
                {line.map((token, k) => (
                  <span key={k} {...getTokenProps({ token })} />
                ))}
              </span>
            ))}
          </pre>
        )}
      </Highlight>
    </figure>
  );
}
