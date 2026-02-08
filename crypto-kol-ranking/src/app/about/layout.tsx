import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About Cryptosensus — How We Detect Crypto Trends",
  description:
    "Cryptosensus tracks data from the smartest degens to surface the tokens that matter. Data-driven insights, not hype.",
  alternates: {
    canonical: "/about",
  },
  openGraph: {
    title: "About Cryptosensus — How We Detect Crypto Trends",
    description:
      "Cryptosensus tracks data from the smartest degens to surface the tokens that matter. Data-driven insights, not hype.",
  },
};

export default function AboutLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
