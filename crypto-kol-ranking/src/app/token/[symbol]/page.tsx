"use client";

import { use, useEffect, useState, useCallback } from "react";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  Copy,
  Check,
  ExternalLink,
  Shield,
  ShieldAlert,
  ShieldX,
  Users,
  MessageSquare,
  TrendingUp,
  TrendingDown,
  Minus,
  Zap,
  Clock,
} from "lucide-react";
import Link from "next/link";
import { Header } from "@/components/layout/Header";
import { FloatingNav } from "@/components/layout/FloatingNav";
import { cn } from "@/lib/utils";

// ─── Types ───────────────────────────────────────────────────────────────────

interface TokenSummary {
  symbol: string;
  score: number;
  score_conviction: number | null;
  score_momentum: number | null;
  mentions: number;
  unique_kols: number;
  sentiment: number;
  conviction_weighted: number;
  trend: "up" | "down" | "stable";
  change_24h: number;
  momentum: number;
  breadth: number;
}

interface TokenSnapshot {
  token_address: string | null;
  price_at_snapshot: number | null;
  market_cap: number | null;
  liquidity_usd: number | null;
  volume_24h: number | null;
  holder_count: number | null;
  top10_holder_pct: number | null;
  price_change_5m: number | null;
  price_change_1h: number | null;
  price_change_6h: number | null;
  price_change_24h: number | null;
  risk_score: number | null;
  has_mint_authority: number | null;
  has_freeze_authority: number | null;
  bundle_detected: number | null;
  bundle_count: number | null;
  bundle_pct: number | null;
  whale_count: number | null;
  whale_total_pct: number | null;
  whale_direction: string | null;
  wash_trading_score: number | null;
  token_age_hours: number | null;
  is_pump_fun: number | null;
  ath_ratio: number | null;
  momentum_direction: string | null;
  price_action_score: number | null;
  top_kols: string[] | null;
  narrative: string | null;
  bubblemaps_score: number | null;
}

interface ApiResponse {
  token: TokenSummary;
  snapshot: TokenSnapshot | null;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatNumber(n: number | null | undefined): string {
  if (n == null || isNaN(n)) return "—";
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  if (n >= 1) return `$${n.toFixed(2)}`;
  if (n > 0) return `$${n.toPrecision(4)}`;
  return "$0";
}

function formatPrice(n: number | null | undefined): string {
  if (n == null || isNaN(n)) return "—";
  if (n >= 1) return `$${n.toFixed(4)}`;
  if (n >= 0.001) return `$${n.toFixed(6)}`;
  return `$${n.toPrecision(4)}`;
}

function formatAge(hours: number | null | undefined): string {
  if (hours == null) return "—";
  if (hours < 1) return `${Math.round(hours * 60)}m`;
  if (hours < 24) return `${Math.round(hours)}h`;
  if (hours < 168) return `${(hours / 24).toFixed(1)}d`;
  return `${Math.round(hours / 168)}w`;
}

function truncateAddress(addr: string): string {
  if (addr.length <= 12) return addr;
  return `${addr.slice(0, 6)}...${addr.slice(-4)}`;
}

function riskLevel(score: number | null): {
  label: string;
  color: string;
  bg: string;
  Icon: typeof Shield;
} {
  if (score == null) return { label: "Unknown", color: "text-zinc-500", bg: "bg-zinc-500/10", Icon: Shield };
  if (score <= 500) return { label: "Low Risk", color: "text-emerald-400", bg: "bg-emerald-400/10", Icon: Shield };
  if (score <= 2000) return { label: "Medium", color: "text-amber-400", bg: "bg-amber-400/10", Icon: ShieldAlert };
  return { label: "High Risk", color: "text-red-400", bg: "bg-red-400/10", Icon: ShieldX };
}

// ─── External links ──────────────────────────────────────────────────────────

const EXTERNAL_LINKS = (address: string) => [
  {
    name: "DexScreener",
    url: `https://dexscreener.com/solana/${address}`,
    color: "hover:text-[#4ade80]",
  },
  {
    name: "Axiom",
    url: `https://axiom.trade/t/${address}`,
    color: "hover:text-[#818cf8]",
  },
  {
    name: "GMGN",
    url: `https://gmgn.ai/sol/token/${address}`,
    color: "hover:text-[#f472b6]",
  },
  {
    name: "Photon",
    url: `https://photon-sol.tinyastro.io/en/lp/${address}`,
    color: "hover:text-[#fb923c]",
  },
  {
    name: "BullX",
    url: `https://bullx.io/terminal?chainId=1399811149&address=${address}`,
    color: "hover:text-[#facc15]",
  },
  {
    name: "Birdeye",
    url: `https://birdeye.so/token/${address}?chain=solana`,
    color: "hover:text-[#38bdf8]",
  },
];

// ─── Sub-components ──────────────────────────────────────────────────────────

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [text]);

  return (
    <button
      onClick={handleCopy}
      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 transition-colors text-sm font-mono text-zinc-400 hover:text-white"
    >
      <span>{truncateAddress(text)}</span>
      {copied ? (
        <Check className="h-3.5 w-3.5 text-emerald-400" />
      ) : (
        <Copy className="h-3.5 w-3.5" />
      )}
    </button>
  );
}

function PriceChangePill({
  value,
  label,
}: {
  value: number | null;
  label: string;
}) {
  if (value == null) return null;
  const isPositive = value > 0;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium tabular-nums",
        isPositive
          ? "bg-emerald-400/10 text-emerald-400"
          : "bg-red-400/10 text-red-400"
      )}
    >
      {isPositive ? "+" : ""}
      {value.toFixed(1)}%
      <span className="text-[10px] opacity-60">{label}</span>
    </span>
  );
}

function MetricCard({
  label,
  value,
  subtext,
}: {
  label: string;
  value: string;
  subtext?: string;
}) {
  return (
    <div className="p-4 rounded-xl border border-white/[0.06] bg-white/[0.02]">
      <p className="text-xs text-zinc-500 mb-1">{label}</p>
      <p className="text-xl font-bold tabular-nums text-white">{value}</p>
      {subtext && (
        <p className="text-[11px] text-zinc-600 mt-0.5">{subtext}</p>
      )}
    </div>
  );
}

function SafetyRow({
  label,
  value,
  safe,
}: {
  label: string;
  value: string;
  safe: boolean | null;
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-white/[0.04] last:border-0">
      <span className="text-xs text-zinc-500">{label}</span>
      <span
        className={cn(
          "text-xs font-medium",
          safe === null
            ? "text-zinc-500"
            : safe
              ? "text-emerald-400"
              : "text-red-400"
        )}
      >
        {value}
      </span>
    </div>
  );
}

// ─── Loading Skeleton ────────────────────────────────────────────────────────

function LoadingSkeleton() {
  return (
    <div className="min-h-screen bg-background">
      <div className="fixed inset-0 bg-grid-pattern pointer-events-none" />
      <Header />
      <FloatingNav />
      <main className="relative z-10 mx-auto max-w-5xl px-6 pt-24 pb-32">
        <div className="mb-8">
          <div className="h-4 w-32 bg-white/5 rounded animate-pulse" />
        </div>
        <div className="mb-8 flex justify-between items-end">
          <div>
            <div className="h-12 w-48 bg-white/5 rounded-lg animate-pulse mb-4" />
            <div className="h-6 w-64 bg-white/5 rounded animate-pulse" />
          </div>
          <div className="h-20 w-24 bg-white/5 rounded-lg animate-pulse" />
        </div>
        <div className="h-10 bg-white/5 rounded-lg animate-pulse mb-6" />
        <div className="h-[400px] bg-white/5 rounded-2xl animate-pulse mb-6" />
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
          {[...Array(4)].map((_, i) => (
            <div
              key={i}
              className="h-24 bg-white/5 rounded-xl animate-pulse"
            />
          ))}
        </div>
        <div className="grid lg:grid-cols-2 gap-4">
          <div className="h-64 bg-white/5 rounded-2xl animate-pulse" />
          <div className="h-64 bg-white/5 rounded-2xl animate-pulse" />
        </div>
      </main>
    </div>
  );
}

// ─── Not Found ───────────────────────────────────────────────────────────────

function TokenNotFound({ symbol }: { symbol: string }) {
  return (
    <div className="min-h-screen bg-background">
      <div className="fixed inset-0 bg-grid-pattern pointer-events-none" />
      <Header />
      <FloatingNav />
      <main className="relative z-10 mx-auto max-w-5xl px-6 pt-24 pb-32">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-sm text-zinc-500 hover:text-white transition-colors mb-12"
        >
          <ArrowLeft className="h-4 w-4" />
          Back
        </Link>
        <div className="text-center py-20">
          <p className="text-6xl font-bold text-zinc-700 mb-4">404</p>
          <p className="text-zinc-500">
            Token <span className="text-white font-medium">${symbol.toUpperCase()}</span> not found
          </p>
        </div>
      </main>
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────

export default function TokenDetailPage({
  params,
}: {
  params: Promise<{ symbol: string }>;
}) {
  const { symbol } = use(params);
  const [data, setData] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch(`/api/token/${encodeURIComponent(symbol)}`);
        if (res.status === 404) {
          setNotFound(true);
          return;
        }
        if (!res.ok) throw new Error("fetch failed");
        const json: ApiResponse = await res.json();
        setData(json);
      } catch {
        setNotFound(true);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [symbol]);

  if (loading) return <LoadingSkeleton />;
  if (notFound || !data) return <TokenNotFound symbol={symbol} />;

  const { token, snapshot } = data;
  const address = snapshot?.token_address;
  const links = address ? EXTERNAL_LINKS(address) : [];
  const risk = riskLevel(snapshot?.risk_score ?? null);
  const TrendIcon =
    token.trend === "up"
      ? TrendingUp
      : token.trend === "down"
        ? TrendingDown
        : Minus;

  const stagger = (i: number) => ({ delay: 0.05 + i * 0.06 });

  return (
    <div className="min-h-screen bg-background">
      <div className="fixed inset-0 bg-grid-pattern pointer-events-none" />
      <Header />
      <FloatingNav />

      <main className="relative z-10 mx-auto max-w-5xl px-6 pt-24 pb-32">
        {/* Back */}
        <motion.div
          initial={{ opacity: 0, x: -12 }}
          animate={{ opacity: 1, x: 0 }}
          className="mb-8"
        >
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-sm text-zinc-500 hover:text-white transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Link>
        </motion.div>

        {/* ─── Section 1: Header ──────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={stagger(0)}
          className="mb-6"
        >
          <div className="flex flex-col sm:flex-row items-start sm:items-end justify-between gap-4 mb-4">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h1 className="text-4xl sm:text-5xl font-bold text-white">
                  {token.symbol}
                </h1>
                {snapshot?.is_pump_fun === 1 && (
                  <span className="px-2 py-0.5 rounded-md text-[10px] font-semibold uppercase tracking-wider bg-orange-500/10 text-orange-400 border border-orange-500/20">
                    Pump.fun
                  </span>
                )}
                {snapshot?.narrative && (
                  <span className="px-2 py-0.5 rounded-md text-[10px] font-medium bg-violet-500/10 text-violet-400 border border-violet-500/20">
                    {snapshot.narrative}
                  </span>
                )}
              </div>

              {/* Price + changes */}
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-lg font-medium text-white tabular-nums">
                  {formatPrice(snapshot?.price_at_snapshot)}
                </span>
                <PriceChangePill value={snapshot?.price_change_5m ?? null} label="5m" />
                <PriceChangePill value={snapshot?.price_change_1h ?? null} label="1h" />
                <PriceChangePill value={snapshot?.price_change_6h ?? null} label="6h" />
                <PriceChangePill value={snapshot?.price_change_24h ?? null} label="24h" />
              </div>

              {/* CA + Age */}
              <div className="flex items-center gap-3 mt-3">
                {address && <CopyButton text={address} />}
                {snapshot?.token_age_hours != null && (
                  <span className="inline-flex items-center gap-1 text-xs text-zinc-500">
                    <Clock className="h-3 w-3" />
                    {formatAge(snapshot.token_age_hours)}
                  </span>
                )}
              </div>
            </div>

            {/* Score */}
            <div className="text-right">
              <p className="text-[10px] text-zinc-500 uppercase tracking-widest mb-1">
                Score
              </p>
              <p className="text-6xl sm:text-7xl font-bold text-white tabular-nums leading-none">
                {token.score}
              </p>
              <div className="flex items-center justify-end gap-1 mt-1">
                <TrendIcon
                  className={cn(
                    "h-4 w-4",
                    token.change_24h > 0
                      ? "text-emerald-400"
                      : token.change_24h < 0
                        ? "text-red-400"
                        : "text-zinc-500"
                  )}
                />
                <span
                  className={cn(
                    "text-sm tabular-nums font-medium",
                    token.change_24h > 0
                      ? "text-emerald-400"
                      : token.change_24h < 0
                        ? "text-red-400"
                        : "text-zinc-500"
                  )}
                >
                  {token.change_24h > 0 ? "+" : ""}
                  {Number(token.change_24h).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* ─── Section 2: Quick Links ─────────────────────────────────── */}
        {address && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={stagger(1)}
            className="flex items-center gap-2 flex-wrap mb-6"
          >
            {/* Primary buy CTA */}
            <a
              href={`https://axiom.trade/t/${address}`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 transition-colors text-sm font-medium"
            >
              <Zap className="h-3.5 w-3.5" />
              Buy on Axiom
            </a>

            {/* Other links */}
            {links
              .filter((l) => l.name !== "Axiom")
              .map((link) => (
                <a
                  key={link.name}
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={cn(
                    "inline-flex items-center gap-1.5 px-3 py-2 rounded-lg bg-white/[0.03] border border-white/[0.06] text-zinc-400 transition-colors text-sm",
                    link.color
                  )}
                >
                  <ExternalLink className="h-3 w-3" />
                  {link.name}
                </a>
              ))}
          </motion.div>
        )}

        {/* ─── Section 3: DexScreener Chart ───────────────────────────── */}
        {address && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={stagger(2)}
            className="mb-6 rounded-2xl border border-white/[0.06] overflow-hidden"
          >
            <iframe
              src={`https://dexscreener.com/solana/${address}?embed=1&theme=dark&info=0`}
              className="w-full h-[420px] bg-black"
              title={`${token.symbol} chart`}
              loading="lazy"
              allowFullScreen
            />
          </motion.div>
        )}

        {/* ─── Section 4: Market Metrics ──────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={stagger(3)}
          className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6"
        >
          <MetricCard
            label="Market Cap"
            value={formatNumber(snapshot?.market_cap)}
          />
          <MetricCard
            label="Liquidity"
            value={formatNumber(snapshot?.liquidity_usd)}
          />
          <MetricCard
            label="Volume 24h"
            value={formatNumber(snapshot?.volume_24h)}
          />
          <MetricCard
            label="Holders"
            value={
              snapshot?.holder_count != null
                ? snapshot.holder_count.toLocaleString()
                : "—"
            }
          />
        </motion.div>

        {/* ─── Section 5 + 6: Safety & KOL Intel (2 columns) ─────────── */}
        <div className="grid lg:grid-cols-2 gap-4">
          {/* Safety & On-Chain */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={stagger(4)}
            className="rounded-2xl border border-white/[0.06] bg-white/[0.01] p-5"
          >
            <div className="flex items-center gap-2 mb-4">
              <risk.Icon className={cn("h-4 w-4", risk.color)} />
              <h2 className="text-sm font-semibold text-white">Safety</h2>
              <span
                className={cn(
                  "ml-auto px-2 py-0.5 rounded-md text-[10px] font-semibold",
                  risk.bg,
                  risk.color
                )}
              >
                {risk.label}
              </span>
            </div>

            <div className="space-y-0">
              <SafetyRow
                label="Mint Authority"
                value={
                  snapshot?.has_mint_authority == null
                    ? "—"
                    : snapshot.has_mint_authority === 0
                      ? "Disabled"
                      : "Active"
                }
                safe={
                  snapshot?.has_mint_authority == null
                    ? null
                    : snapshot.has_mint_authority === 0
                }
              />
              <SafetyRow
                label="Freeze Authority"
                value={
                  snapshot?.has_freeze_authority == null
                    ? "—"
                    : snapshot.has_freeze_authority === 0
                      ? "Disabled"
                      : "Active"
                }
                safe={
                  snapshot?.has_freeze_authority == null
                    ? null
                    : snapshot.has_freeze_authority === 0
                }
              />
              <SafetyRow
                label="Top 10 Holders"
                value={
                  snapshot?.top10_holder_pct != null
                    ? `${Number(snapshot.top10_holder_pct).toFixed(1)}%`
                    : "—"
                }
                safe={
                  snapshot?.top10_holder_pct != null
                    ? Number(snapshot.top10_holder_pct) < 50
                    : null
                }
              />
              <SafetyRow
                label="Bundles"
                value={
                  snapshot?.bundle_detected == null
                    ? "—"
                    : snapshot.bundle_detected
                      ? `Detected (${snapshot.bundle_count ?? "?"})`
                      : "None"
                }
                safe={
                  snapshot?.bundle_detected == null
                    ? null
                    : !snapshot.bundle_detected
                }
              />
              <SafetyRow
                label="Whale Direction"
                value={snapshot?.whale_direction ?? "—"}
                safe={
                  snapshot?.whale_direction == null
                    ? null
                    : snapshot.whale_direction === "accumulating" ||
                      snapshot.whale_direction === "holding"
                }
              />
              {snapshot?.whale_total_pct != null && (
                <SafetyRow
                  label="Whale Holdings"
                  value={`${Number(snapshot.whale_total_pct).toFixed(1)}%`}
                  safe={Number(snapshot.whale_total_pct) < 30}
                />
              )}
              {snapshot?.wash_trading_score != null && (
                <SafetyRow
                  label="Wash Trading"
                  value={
                    Number(snapshot.wash_trading_score) > 0.5
                      ? "Suspected"
                      : "Clean"
                  }
                  safe={Number(snapshot.wash_trading_score) <= 0.5}
                />
              )}
            </div>
          </motion.div>

          {/* KOL Intelligence */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={stagger(5)}
            className="rounded-2xl border border-white/[0.06] bg-white/[0.01] p-5"
          >
            <h2 className="text-sm font-semibold text-white mb-4">
              KOL Intelligence
            </h2>

            {/* Stats row */}
            <div className="grid grid-cols-3 gap-3 mb-4">
              <div className="text-center">
                <div className="flex items-center justify-center gap-1 mb-1">
                  <MessageSquare className="h-3 w-3 text-blue-400" />
                </div>
                <p className="text-xl font-bold tabular-nums">
                  {token.mentions}
                </p>
                <p className="text-[10px] text-zinc-500">Mentions</p>
              </div>
              <div className="text-center">
                <div className="flex items-center justify-center gap-1 mb-1">
                  <Users className="h-3 w-3 text-green-400" />
                </div>
                <p className="text-xl font-bold tabular-nums">
                  {token.unique_kols}
                </p>
                <p className="text-[10px] text-zinc-500">KOLs</p>
              </div>
              <div className="text-center">
                <div className="flex items-center justify-center gap-1 mb-1">
                  <TrendingUp className="h-3 w-3 text-purple-400" />
                </div>
                <p className="text-xl font-bold tabular-nums">
                  {Number(token.conviction_weighted).toFixed(1)}
                </p>
                <p className="text-[10px] text-zinc-500">Conviction</p>
              </div>
            </div>

            {/* Sentiment bar */}
            <div className="mb-4">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-zinc-500">Sentiment</span>
                <span className="text-xs font-medium tabular-nums">
                  {(Number(token.sentiment) * 100).toFixed(0)}%
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{
                    width: `${Math.max(0, Math.min(100, ((Number(token.sentiment) + 1) / 2) * 100))}%`,
                  }}
                  transition={{ duration: 0.8, delay: 0.4 }}
                  className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-emerald-300"
                />
              </div>
            </div>

            {/* Momentum + Price Action */}
            <div className="flex items-center gap-2 mb-4 flex-wrap">
              {snapshot?.momentum_direction && (
                <span
                  className={cn(
                    "px-2 py-0.5 rounded-md text-[10px] font-medium border",
                    snapshot.momentum_direction === "bouncing"
                      ? "bg-emerald-400/10 text-emerald-400 border-emerald-400/20"
                      : snapshot.momentum_direction === "pumping"
                        ? "bg-red-400/10 text-red-400 border-red-400/20"
                        : snapshot.momentum_direction === "dying"
                          ? "bg-red-400/10 text-red-400 border-red-400/20"
                          : "bg-zinc-500/10 text-zinc-400 border-zinc-500/20"
                  )}
                >
                  {snapshot.momentum_direction}
                </span>
              )}
              {snapshot?.price_action_score != null && (
                <span className="text-[10px] text-zinc-500">
                  PA Score:{" "}
                  <span className="text-zinc-300 font-medium">
                    {(Number(snapshot.price_action_score) * 100).toFixed(0)}
                  </span>
                </span>
              )}
              {snapshot?.ath_ratio != null && (
                <span className="text-[10px] text-zinc-500">
                  ATH Ratio:{" "}
                  <span className="text-zinc-300 font-medium">
                    {(Number(snapshot.ath_ratio) * 100).toFixed(0)}%
                  </span>
                </span>
              )}
            </div>

            {/* Top KOLs list */}
            {snapshot?.top_kols &&
              Array.isArray(snapshot.top_kols) &&
              snapshot.top_kols.length > 0 && (
                <div>
                  <p className="text-xs text-zinc-500 mb-2">
                    Top KOLs mentioning
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {snapshot.top_kols.slice(0, 8).map((kol) => (
                      <span
                        key={typeof kol === "string" ? kol : JSON.stringify(kol)}
                        className="px-2 py-0.5 rounded-md text-[11px] bg-white/[0.04] border border-white/[0.06] text-zinc-400"
                      >
                        {typeof kol === "string" ? kol : String(kol)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
          </motion.div>
        </div>
      </main>
    </div>
  );
}
