import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description:
    "How Cryptosensus collects, uses, and protects your data. Read our privacy policy.",
  alternates: {
    canonical: "/privacy",
  },
  openGraph: {
    title: "Privacy Policy — Cryptosensus",
    description:
      "How Cryptosensus collects, uses, and protects your data. Read our privacy policy.",
  },
};

export default function PrivacyLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
