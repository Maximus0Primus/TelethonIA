import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
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

export const metadata: Metadata = {
  title: "Cryptosensus",
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
  ],
  authors: [{ name: "Cryptosensus" }],
  openGraph: {
    title: "Cryptosensus",
    description:
      "Data-driven insights from 50+ crypto KOLs",
    type: "website",
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
        className={`${inter.variable} ${jetbrainsMono.variable} font-sans antialiased min-h-screen bg-background`}
      >
        {children}
      </body>
    </html>
  );
}
