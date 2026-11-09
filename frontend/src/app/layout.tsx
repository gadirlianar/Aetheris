import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aetheris — Claim-Level Verifiable Multi-Agent Deep Research Engine",
  description:
    "Autonomous scientific research engine with 3-way Hybrid + GraphRAG retrieval, 3-pass mechanical quote grounding, and adversarial Skeptic verification.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#070b14] text-slate-100 antialiased selection:bg-cyan-500/30 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}
