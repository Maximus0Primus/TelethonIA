import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Terms of Service",
  description:
    "Terms and conditions for using the Cryptosensus platform and its services.",
  alternates: {
    canonical: "/terms",
  },
  openGraph: {
    title: "Terms of Service — Cryptosensus",
    description:
      "Terms and conditions for using the Cryptosensus platform and its services.",
  },
};

export default function TermsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
