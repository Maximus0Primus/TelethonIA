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
    default: "Cryptosensus — Real-Time Crypto Lowcap Buy Score",
    template: "%s | Cryptosensus",
  },
  description:
    "Cryptosensus brings clarity to the noise. Real-time crypto lowcap buy scores to surface the tokens that matter.",
  keywords: [
    "crypto",
    "cryptosensus",
    "token",
    "ranking",
    "sentiment",
    "memecoin",
    "crypto sentiment",
    "memecoin ranking",
    "crypto analytics",
    "crypto trading signals",
  ],
  authors: [{ name: "Cryptosensus" }],
  openGraph: {
    title: "Cryptosensus — Real-Time Crypto Lowcap Buy Score",
    description:
      "Real-time crypto lowcap buy scores. Data-driven insights to surface the tokens that matter.",
    type: "website",
    siteName: "Cryptosensus",
  },
  twitter: {
    card: "summary_large_image",
    title: "Cryptosensus — Real-Time Crypto Lowcap Buy Score",
    description:
      "Real-time crypto lowcap buy scores. Data-driven insights to surface the tokens that matter.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // JSON-LD structured data — static content only, no user input
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "WebApplication",
    name: "Cryptosensus",
    url: BASE_URL,
    description:
      "Real-time crypto lowcap buy scores to surface the tokens that matter.",
    applicationCategory: "FinanceApplication",
    operatingSystem: "Web",
    offers: {
      "@type": "Offer",
      price: "0",
      priceCurrency: "USD",
    },
  };

  return (
    <html lang="en">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} font-sans antialiased min-h-screen bg-background overflow-x-hidden`}
      >
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
        <AudioProvider>{children}</AudioProvider>
      </body>
    </html>
  );
}
