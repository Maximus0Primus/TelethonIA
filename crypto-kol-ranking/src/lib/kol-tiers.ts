export interface KolTierInfo {
  tier: "S" | "A";
  conviction: number;
}

export const KOL_TIERS: Record<string, KolTierInfo> = {
  // S-TIER (weight 2.0, conviction 10)
  archercallz: { tier: "S", conviction: 10 },
  MoonsCallz: { tier: "S", conviction: 10 },
  Luca_Apes: { tier: "S", conviction: 10 },
  donniesdegen: { tier: "S", conviction: 10 },
  legerlegends: { tier: "S", conviction: 10 },
  ghastlygems: { tier: "S", conviction: 10 },
  certifiedprintor: { tier: "S", conviction: 10 },
  bounty_journal: { tier: "S", conviction: 10 },
  degenncabal: { tier: "S", conviction: 10 },
  eveesL: { tier: "S", conviction: 10 },
  MaybachGambleCalls: { tier: "S", conviction: 10 },
  MaybachCalls: { tier: "S", conviction: 10 },
  darkocalls: { tier: "S", conviction: 10 },
  // A-TIER (weight 1.0, conviction 7)
  BrodyCalls: { tier: "A", conviction: 7 },
  explorer_gems: { tier: "A", conviction: 7 },
  missorplays: { tier: "A", conviction: 7 },
  ramcalls: { tier: "A", conviction: 7 },
  snoopsalpha: { tier: "A", conviction: 7 },
  slingoorioyaps: { tier: "A", conviction: 7 },
  ALSTEIN_GEMCLUB: { tier: "A", conviction: 7 },
  wuziemakesmoney: { tier: "A", conviction: 7 },
  letswinallgems: { tier: "A", conviction: 7 },
  dylansdegens: { tier: "A", conviction: 7 },
  BossmanCallsOfficial: { tier: "A", conviction: 7 },
  menacedegendungeon: { tier: "A", conviction: 7 },
  arcanegems: { tier: "A", conviction: 7 },
  PumpItCabal: { tier: "A", conviction: 7 },
  MarcellsFightclub: { tier: "A", conviction: 7 },
  x666calls: { tier: "A", conviction: 7 },
  dylansdirtydiary: { tier: "A", conviction: 7 },
  invacooksclub: { tier: "A", conviction: 7 },
  leoclub168c: { tier: "A", conviction: 7 },
  caniscooks: { tier: "A", conviction: 7 },
  maritocalls: { tier: "A", conviction: 7 },
  PowsGemCalls: { tier: "A", conviction: 7 },
  waldosalpha: { tier: "A", conviction: 7 },
  LevisAlpha: { tier: "A", conviction: 7 },
  eunicalls: { tier: "A", conviction: 7 },
  spidersjournal: { tier: "A", conviction: 7 },
  marcellcooks: { tier: "A", conviction: 7 },
  DegenSeals: { tier: "A", conviction: 7 },
  KittysKasino: { tier: "A", conviction: 7 },
  Archerrgambles: { tier: "A", conviction: 7 },
  fakepumpsbynumer0: { tier: "A", conviction: 7 },
  robogems: { tier: "A", conviction: 7 },
  shahlito: { tier: "A", conviction: 7 },
  CryptoChefCooks: { tier: "A", conviction: 7 },
  LittleMustachoCalls: { tier: "A", conviction: 7 },
  OnyxxGems: { tier: "A", conviction: 7 },
  pantherjournal: { tier: "A", conviction: 7 },
  CSCalls: { tier: "A", conviction: 7 },
  kweensjournal: { tier: "A", conviction: 7 },
  lollycalls: { tier: "A", conviction: 7 },
  CatfishcallsbyPoe: { tier: "A", conviction: 7 },
  CarnagecallsGambles: { tier: "A", conviction: 7 },
  ChairmanDN1: { tier: "A", conviction: 7 },
  NisoksChadHouse: { tier: "A", conviction: 7 },
  sadcatgamble: { tier: "A", conviction: 7 },
  shmooscasino: { tier: "A", conviction: 7 },
  AnimeGems: { tier: "A", conviction: 7 },
  veigarcalls: { tier: "A", conviction: 7 },
  papicall: { tier: "A", conviction: 7 },
};

// Normalized KOL scores from kol_scorer.py (0.1 - 3.0)
// First-call-per-token methodology: each (KOL, token) counted once.
// Score = hit_rate / baseline. 1.0 = average, >1.0 = above avg.
// Updated: 2026-02-14 (v17 first-call-per-token, multi-horizon)
export const KOL_SCORES: Record<string, number> = {
  ChairmanDN1: 0.1,
  ghastlygems: 0.833,
  eveesL: 0.567,
  AnimeGems: 0.811,
  MaybachGambleCalls: 0.661,
  veigarcalls: 1.859,
  LevisAlpha: 2.335,
  BrodyCalls: 1.316,
  CryptoChefCooks: 1.519,
  BossmanCallsOfficial: 0.665,
  spidersjournal: 0.947,
  slingoorioyaps: 0.75,
  bounty_journal: 0.1,
  letswinallgems: 1.527,
  legerlegends: 2.35,
  explorer_gems: 0.1,
  DegenSeals: 0.82,
  menacedegendungeon: 0.1,
  Archerrgambles: 2.307,
  ALSTEIN_GEMCLUB: 1.145,
  PowsGemCalls: 1.333,
  MaybachCalls: 0.1,
  CarnagecallsGambles: 1.556,
};

export const S_TIER_COUNT = Object.values(KOL_TIERS).filter(
  (k) => k.tier === "S"
).length;

export const TOTAL_KOLS = Object.keys(KOL_TIERS).length;
