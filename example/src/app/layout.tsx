import type { Metadata } from "next";
import { Analytics } from "@/components/analytics";
import { TopNav } from "@/components/topnav";
import "./globals.css";


export const metadata: Metadata = {
  title: "smokey-fluid-cursor — live demo",
  description: "WebGL fluid-simulation cursor trails for any website.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>
        <TopNav pkg="smokey-fluid-cursor" />
        {children}
        <Analytics packageName="smokey-fluid-cursor" />
      </body>
    </html>
  );
}
