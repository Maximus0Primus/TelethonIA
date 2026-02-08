import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { AudioProvider } from "@/components/AudioProvider";
import { Footer } from "@/components/layout/Footer";
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
    default: "Cryptosensus — Real-Time Lowcap Crypto Buy Indicator",
    template: "%s | Cryptosensus",
  },
  description:
    "Cryptosensus brings clarity to the noise. We score lowcap tokens so you know what to buy before the crowd does. Stop guessing. Start winning.",
  authors: [{ name: "Cryptosensus" }],
  alternates: {
    canonical: BASE_URL,
  },
  openGraph: {
    title: "Cryptosensus — Real-Time Lowcap Crypto Buy Indicator",
    description:
      "Cryptosensus brings clarity to the noise. We score lowcap tokens so you know what to buy before the crowd does. Stop guessing. Start winning.",
    url: BASE_URL,
    type: "website",
    siteName: "Cryptosensus",
    images: [
      {
        url: "/opengraph-image",
        width: 1200,
        height: 630,
        alt: "Cryptosensus — Real-Time Lowcap Crypto Buy Indicator",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    creator: "@Maximus0Primus",
    title: "Cryptosensus — Real-Time Lowcap Crypto Buy Indicator",
    description:
      "Cryptosensus brings clarity to the noise. We score lowcap tokens so you know what to buy before the crowd does. Stop guessing. Start winning.",
    images: ["/opengraph-image"],
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
      "Cryptosensus brings clarity to the noise. We score lowcap tokens so you know what to buy before the crowd does. Stop guessing. Start winning.",
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
        <AudioProvider>
          {children}
          <Footer />
        </AudioProvider>
      </body>
    </html>
  );
}
