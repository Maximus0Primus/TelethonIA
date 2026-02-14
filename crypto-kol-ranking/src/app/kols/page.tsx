import { KolLeaderboard } from "@/components/kols/KolLeaderboard";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "KOL Leaderboard | Cryptosensus",
  description:
    "Track the performance of 59 crypto KOLs. See win rates, call accuracy, and tier rankings updated in real-time.",
};

export default function KolsPage() {
  return (
    <main className="min-h-screen pt-12 pb-32 px-4">
      <div className="max-w-4xl mx-auto mb-8 text-center">
        <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-white mb-2">
          KOL Leaderboard
        </h1>
        <p className="text-sm text-white/40">
          Performance tracking for 59 monitored Telegram KOLs
        </p>
      </div>

      <KolLeaderboard />
    </main>
  );
}
