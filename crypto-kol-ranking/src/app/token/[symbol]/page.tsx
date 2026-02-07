"use client";

import { use } from "react";
import { motion } from "framer-motion";
import { ArrowLeft, MessageSquare, Users, TrendingUp, TrendingDown, Minus } from "lucide-react";
import Link from "next/link";
import { Header } from "@/components/layout/Header";
import { FloatingNav } from "@/components/layout/FloatingNav";
import { cn } from "@/lib/utils";

// Mock data for token details
const mockTokenDetail = {
  symbol: "$PEPE",
  score: 87,
  mentions: 234,
  uniqueKols: 23,
  sentiment: 0.72,
  trend: "up" as const,
  change24h: 124.5,
  change7d: 342.1,
  momentum: 0.85,
  breadth: 0.76,
  convictionWeighted: 8.4,
  topKols: [
    { name: "CryptoKing", conviction: 10, mentions: 12, sentiment: 0.9 },
    { name: "MoonHunter", conviction: 9, mentions: 8, sentiment: 0.85 },
    { name: "AlphaSeeker", conviction: 9, mentions: 6, sentiment: 0.78 },
    { name: "GemFinder", conviction: 8, mentions: 5, sentiment: 0.82 },
    { name: "WhaleWatch", conviction: 8, mentions: 4, sentiment: 0.65 },
  ],
  recentMentions: [
    { kol: "CryptoKing", text: "Strong accumulation signals on $PEPE...", time: "2h ago", sentiment: 0.9 },
    { kol: "MoonHunter", text: "$PEPE looking ready for another leg up", time: "4h ago", sentiment: 0.85 },
    { kol: "AlphaSeeker", text: "Adding more $PEPE to the portfolio", time: "6h ago", sentiment: 0.78 },
  ],
};

export default function TokenDetailPage({
  params,
}: {
  params: Promise<{ symbol: string }>;
}) {
  const { symbol } = use(params);
  const token = mockTokenDetail;

  const TrendIcon = token.trend === "up" ? TrendingUp : token.trend === "down" ? TrendingDown : Minus;

  return (
    <div className="min-h-screen bg-background">
      {/* Background grid */}
      <div className="fixed inset-0 bg-grid-pattern pointer-events-none" />

      <Header />
      <FloatingNav />

      <main className="relative z-10 mx-auto max-w-5xl px-6 pt-24 pb-32">
        {/* Back button */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="mb-8"
        >
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-white transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Tokens
          </Link>
        </motion.div>

        {/* Token Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-12"
        >
          <div className="flex flex-col sm:flex-row items-start sm:items-end justify-between gap-6">
            <div>
              <h1 className="text-5xl sm:text-7xl font-bold text-white mb-4">
                {token.symbol}
              </h1>
              <div className="flex items-center gap-4">
                <div
                  className={cn(
                    "flex items-center gap-2 text-lg font-medium",
                    token.change24h > 0 ? "text-bullish" : token.change24h < 0 ? "text-bearish" : "text-muted-foreground"
                  )}
                >
                  <TrendIcon className="h-5 w-5" />
                  <span>
                    {token.change24h > 0 ? "+" : ""}
                    {token.change24h.toFixed(1)}%
                  </span>
                  <span className="text-muted-foreground text-sm">24h</span>
                </div>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-muted-foreground mb-2">Cryptosensus Score</p>
              <p className="text-6xl sm:text-8xl font-bold text-white tabular-nums">
                {token.score}
              </p>
            </div>
          </div>
        </motion.div>

        {/* Stats Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-12"
        >
          <div className="p-5 rounded-2xl border border-card-border bg-card">
            <div className="flex items-center gap-3 mb-3">
              <div className="rounded-lg bg-blue-500/10 p-2">
                <MessageSquare className="h-5 w-5 text-blue-400" />
              </div>
              <span className="text-sm text-muted-foreground">Mentions</span>
            </div>
            <p className="text-3xl font-bold tabular-nums">{token.mentions}</p>
          </div>

          <div className="p-5 rounded-2xl border border-card-border bg-card">
            <div className="flex items-center gap-3 mb-3">
              <div className="rounded-lg bg-green-500/10 p-2">
                <Users className="h-5 w-5 text-green-400" />
              </div>
              <span className="text-sm text-muted-foreground">Unique KOLs</span>
            </div>
            <p className="text-3xl font-bold tabular-nums">{token.uniqueKols}</p>
          </div>

          <div className="p-5 rounded-2xl border border-card-border bg-card">
            <div className="flex items-center gap-3 mb-3">
              <div className="rounded-lg bg-purple-500/10 p-2">
                <TrendingUp className="h-5 w-5 text-purple-400" />
              </div>
              <span className="text-sm text-muted-foreground">Momentum</span>
            </div>
            <p className="text-3xl font-bold tabular-nums">{(token.momentum * 100).toFixed(0)}%</p>
          </div>

          <div className="p-5 rounded-2xl border border-card-border bg-card">
            <div className="flex items-center gap-3 mb-3">
              <span className="text-sm text-muted-foreground">Sentiment</span>
            </div>
            <div className="flex items-center gap-3">
              <p className="text-3xl font-bold tabular-nums">{(token.sentiment * 100).toFixed(0)}%</p>
              <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${token.sentiment * 100}%` }}
                  transition={{ duration: 0.8, delay: 0.5 }}
                  className="h-full bg-gradient-to-r from-green-500 to-emerald-400 rounded-full"
                />
              </div>
            </div>
          </div>
        </motion.div>

        {/* Two Column Layout */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Top KOLs */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="rounded-2xl border border-card-border bg-card p-6"
          >
            <h2 className="text-lg font-semibold mb-6">Top KOLs</h2>
            <div className="space-y-4">
              {token.topKols.map((kol, index) => (
                <div
                  key={kol.name}
                  className="flex items-center justify-between py-3 border-b border-card-border/50 last:border-0"
                >
                  <div className="flex items-center gap-4">
                    <span className="text-sm text-muted-foreground font-mono w-6">
                      {index + 1}
                    </span>
                    <div className="h-9 w-9 rounded-full bg-muted flex items-center justify-center text-sm font-bold">
                      {kol.name.slice(0, 2).toUpperCase()}
                    </div>
                    <div>
                      <p className="font-medium">{kol.name}</p>
                      <p className="text-xs text-muted-foreground">
                        Conviction {kol.conviction}/10
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-mono text-sm">{kol.mentions} mentions</p>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Recent Mentions */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="rounded-2xl border border-card-border bg-card p-6"
          >
            <h2 className="text-lg font-semibold mb-6">Recent Mentions</h2>
            <div className="space-y-4">
              {token.recentMentions.map((mention, index) => (
                <div
                  key={index}
                  className="p-4 rounded-xl bg-muted/50 border border-card-border/50"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-sm">{mention.kol}</span>
                    <span className="text-xs text-muted-foreground">{mention.time}</span>
                  </div>
                  <p className="text-sm text-muted-foreground line-clamp-2">
                    {mention.text}
                  </p>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Score Breakdown */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-6 rounded-2xl border border-card-border bg-card p-6"
        >
          <h2 className="text-lg font-semibold mb-6">Score Breakdown</h2>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-6">
            {[
              { label: "KOL Consensus", value: 0.35, weight: "35%" },
              { label: "Sentiment", value: (token.sentiment + 1) / 2, weight: "25%" },
              { label: "Conviction", value: token.convictionWeighted / 10, weight: "20%" },
              { label: "Momentum", value: (token.momentum + 1) / 2, weight: "15%" },
              { label: "Breadth", value: token.breadth, weight: "5%" },
            ].map((component) => (
              <div key={component.label}>
                <p className="text-xs text-muted-foreground mb-2">
                  {component.label} <span className="opacity-60">({component.weight})</span>
                </p>
                <div className="relative h-1.5 rounded-full bg-muted overflow-hidden mb-2">
                  <motion.div
                    className="absolute inset-y-0 left-0 rounded-full bg-white"
                    initial={{ width: 0 }}
                    animate={{ width: `${component.value * 100}%` }}
                    transition={{ duration: 0.8, delay: 0.6 }}
                  />
                </div>
                <p className="text-lg font-bold tabular-nums">
                  {(component.value * 100).toFixed(0)}
                </p>
              </div>
            ))}
          </div>
        </motion.div>
      </main>
    </div>
  );
}
