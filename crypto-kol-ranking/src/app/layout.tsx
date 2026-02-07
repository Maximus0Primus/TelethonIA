import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { AudioProvider } from "@/components/AudioProvider";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
});

const BASE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://cryptosensus.org";

export const metadata: Metadata = {
  metadataBase: new URL(BASE_URL),
  title: {
    default: "Cryptosensus — Crypto KOL Sentiment Rankings",
    template: "%s | Cryptosensus",
  },
  description:
    "Cryptosensus brings clarity to the noise. We track sentiment from 50+ crypto KOLs to surface the tokens that matter.",
  keywords: [
    "crypto",
    "cryptosensus",
    "token",
    "ranking",
    "sentiment",
    "memecoin",
    "KOL",
    "crypto sentiment",
    "memecoin ranking",
    "crypto analytics",
  ],
  authors: [{ name: "Cryptosensus" }],
  openGraph: {
    title: "Cryptosensus — Crypto KOL Sentiment Rankings",
    description:
      "Data-driven insights from 50+ crypto KOLs. Real-time memecoin sentiment tracking.",
    type: "website",
    siteName: "Cryptosensus",
  },
  twitter: {
    card: "summary_large_image",
    title: "Cryptosensus",
    description:
      "Data-driven insights from 50+ crypto KOLs",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} font-sans antialiased min-h-screen bg-background overflow-x-hidden`}
      >
        <AudioProvider>{children}</AudioProvider>
      </body>
    </html>
  );
}
