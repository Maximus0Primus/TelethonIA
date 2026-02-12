import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Disclaimer",
  description:
    "Important disclaimers about using Cryptosensus crypto onchain analysis. Not financial advice.",
  alternates: {
    canonical: "/disclaimer",
  },
  openGraph: {
    title: "Disclaimer — Cryptosensus",
    description:
      "Important disclaimers about using Cryptosensus crypto onchain analysis. Not financial advice.",
  },
};

export default function DisclaimerLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
