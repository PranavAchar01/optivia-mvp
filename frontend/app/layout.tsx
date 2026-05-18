import type { Metadata } from "next";
import { Geist_Mono } from "next/font/google";
import "./globals.css";

const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
});

export const metadata: Metadata = {
  title: "Optivia",
  description: "Pre-execution optimization layer for agentic coding CLIs",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`dark ${geistMono.variable}`}>
      <body className="min-h-screen bg-background" suppressHydrationWarning>{children}</body>
    </html>
  );
}
