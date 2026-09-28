"use client";

import { useEffect } from "react";
import Script from "next/script";

const GA_ID = process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID;
const APTABASE_KEY = process.env.NEXT_PUBLIC_APTABASE_KEY;

/**
 * Analytics for the demo site only.
 *
 * Nothing here ships inside the npm package: the published library sends no
 * telemetry of any kind. This runs on the GitHub Pages demo so we can see
 * which examples people actually use.
 */
export function Analytics({ packageName }: { packageName: string }) {
  useEffect(() => {
    if (!APTABASE_KEY) return;

    let cancelled = false;
    // Imported lazily so the analytics bundle never blocks first paint, and so
    // the demo still renders if the request is blocked.
    import("@aptabase/web")
      .then(({ init, trackEvent }) => {
        if (cancelled) return;
        init(APTABASE_KEY);
        trackEvent("demo_viewed", { package: packageName });
      })
      .catch(() => {
        /* an ad blocker or offline visitor — the demo must still work */
      });

    return () => {
      cancelled = true;
    };
  }, [packageName]);

  if (!GA_ID) return null;

  return (
    <>
      <Script
        src={`https://www.googletagmanager.com/gtag/js?id=${GA_ID}`}
        strategy="afterInteractive"
      />
      <Script id="ga-init" strategy="afterInteractive">
        {`
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
          gtag('config', '${GA_ID}');
        `}
      </Script>
    </>
  );
}

/** Fire-and-forget event helper for demo interactions. */
export async function track(name: string, props?: Record<string, string | number | boolean>) {
  if (!APTABASE_KEY) return;
  try {
    const { trackEvent } = await import("@aptabase/web");
    trackEvent(name, props);
  } catch {
    /* ignore */
  }
}
