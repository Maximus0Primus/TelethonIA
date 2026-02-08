"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { GlitchText } from "./GlitchText";
import { PasswordGate } from "./PasswordGate";
import { ContactModal } from "./ContactModal";
import { useAudio } from "@/components/AudioProvider";


const STORAGE_KEY = "cryptosensus_intro_seen_v4";
const ACCESS_TOKEN_KEY = "cryptosensus_access_token";

const TITLE_CLASSES =
  "text-[clamp(2.75rem,10vw,8rem)] font-bold tracking-tight text-white select-none whitespace-nowrap";

// step 1: "Enter" fullglitch  → 600ms
// step 2: pause               → 180ms
// step 3: "The" fullglitch    → 500ms
// step 4: pause               → 150ms
// step 5: "Crypto" reveal + "sensus" fade → 1800ms fallback
// step 6: ambient (final)
const STEP_TIMINGS: Record<number, number> = {
  1: 600,
  2: 180,
  3: 500,
  4: 150,
  5: 1800,
};

const FEATURES = [
  { title: "Clarity From Noise", desc: "We track what the smartest degens are talking about" },
  { title: "Data-Driven Insights", desc: "Not hype. Not shills. Just data that matters" },
  { title: "Real Traction Detection", desc: "Our algorithm detects which tokens are gaining momentum" },
  { title: "Conviction Scoring", desc: "Each token gets a score from 0 to 100" },
  { title: "Safety Analysis", desc: "Rug detection, holder distribution, on-chain checks" },
  { title: "Always Up To Date", desc: "Fully automated, updated around the clock" },
];

// Static preview data — based on recent real rankings
const PREVIEW_TOKENS = [
  { symbol: "$TESLA", score: 40, rank: 1 },
  { symbol: "$YEE", score: 39, rank: 2 },
  { symbol: "$PETAH", score: 37, rank: 3 },
  { symbol: "$PENGUIN", score: 36, rank: 4 },
  { symbol: "$WAR", score: 35, rank: 5 },
  { symbol: "$BIGTROUT", score: 35, rank: 6 },
  { symbol: "$SHARK", score: 34, rank: 7 },
  { symbol: "$WHITEWHALE", score: 34, rank: 8 },
];

const PODIUM_COLORS: Record<number, { border: string; score: string; badgeBg: string; badge: string }> = {
  1: { border: "rgba(201,169,98,0.3)", score: "#e2c878", badgeBg: "rgba(201,169,98,0.1)", badge: "#c9a962" },
  2: { border: "rgba(168,174,184,0.25)", score: "#bcc2cc", badgeBg: "rgba(168,174,184,0.1)", badge: "#a8aeb8" },
  3: { border: "rgba(161,122,86,0.25)", score: "#c49b74", badgeBg: "rgba(161,122,86,0.1)", badge: "#a17a56" },
};

interface HeroSectionProps {
  onIntroComplete?: () => void;
}

// Scroll sentinel config: each fires once, dispatching permanent cells
const SCROLL_SENTINELS = [
  { id: "sentinel-1", count: 4, minIntensity: 0.08, maxIntensity: 0.2 },
  { id: "sentinel-2", count: 6, minIntensity: 0.1, maxIntensity: 0.3 },
  { id: "sentinel-3", count: 10, minIntensity: 0.15, maxIntensity: 0.4 },
  { id: "sentinel-4", count: 14, minIntensity: 0.2, maxIntensity: 0.5 },
  { id: "sentinel-5", count: 20, minIntensity: 0.25, maxIntensity: 0.6 },
];

export function HeroSection({ onIntroComplete }: HeroSectionProps) {
  const [mounted, setMounted] = useState(false);
  const [hasSeenIntro, setHasSeenIntro] = useState(false);
  const [step, setStep] = useState(0);

  // Auth states
  const [authChecked, setAuthChecked] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  const { playStart, playLoop, stopLoop, unlock, unlocked } = useAudio();

  // Scroll sentinel refs
  const sentinelRefs = useRef<(HTMLDivElement | null)[]>([]);
  const activeSentinels = useRef<Set<string>>(new Set());

  // Cursor sound tooltip
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });
  const [cursorVisible, setCursorVisible] = useState(false);

  const handleLandingMouseMove = useCallback((e: React.MouseEvent) => {
    setCursorPos({ x: e.clientX, y: e.clientY });
    if (!cursorVisible) setCursorVisible(true);
  }, [cursorVisible]);

  const handleLandingClick = useCallback(() => {
    if (unlocked) return;
    unlock();
    setTimeout(() => playLoop(), 100);
  }, [unlocked, unlock, playLoop]);

  // Mount + localStorage check + token verification
  useEffect(() => {
    setMounted(true);

    const seen = localStorage.getItem(STORAGE_KEY);
    if (seen) {
      setHasSeenIntro(true);
    }

    // Verify stored token
    const token = localStorage.getItem(ACCESS_TOKEN_KEY);
    if (!token) {
      setAuthChecked(true);
      return;
    }

    fetch("/api/auth/verify-token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.valid) {
          setIsAuthenticated(true);
          setHasSeenIntro(true);
        } else {
          localStorage.removeItem(ACCESS_TOKEN_KEY);
        }
      })
      .catch(() => {
        setIsAuthenticated(true);
        setHasSeenIntro(true);
      })
      .finally(() => {
        setAuthChecked(true);
      });
  }, []);

  // Password verified — store token, unlock audio, begin intro sequence
  const handlePasswordSuccess = (token: string) => {
    localStorage.setItem(ACCESS_TOKEN_KEY, token);
    setIsAuthenticated(true);
    if (!unlocked) unlock();
    stopLoop();
    playStart();
    setStep(1);
  };

  // Progressive scroll-triggered permanent grid breaks (bidirectional)
  useEffect(() => {
    if (!mounted || !authChecked || hasSeenIntro || step !== 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const id = entry.target.getAttribute("data-sentinel");
          if (!id) continue;

          if (entry.isIntersecting && !activeSentinels.current.has(id)) {
            // Sentinel entered viewport → add cells
            activeSentinels.current.add(id);
            const config = SCROLL_SENTINELS.find((s) => s.id === id);
            if (config) {
              window.dispatchEvent(
                new CustomEvent("grid-break-permanent", {
                  detail: {
                    count: config.count,
                    minIntensity: config.minIntensity,
                    maxIntensity: config.maxIntensity,
                    group: id,
                  },
                })
              );
            }
          } else if (!entry.isIntersecting && activeSentinels.current.has(id)) {
            // Sentinel left viewport — only clear if it went BELOW (user scrolled up)
            // If sentinel.bottom < 0 → it's above viewport → user scrolled down → keep cells
            const isAboveViewport = entry.boundingClientRect.bottom < 0;
            if (!isAboveViewport) {
              activeSentinels.current.delete(id);
              window.dispatchEvent(
                new CustomEvent("grid-clear-group", { detail: { group: id } })
              );
            }
          }
        }
      },
      { threshold: 0 }
    );

    for (const el of sentinelRefs.current) {
      if (el) observer.observe(el);
    }

    return () => observer.disconnect();
  }, [mounted, authChecked, hasSeenIntro, step]);

  // Returning visitor: click anywhere to start ambient loop
  const handleUnlockLoop = () => {
    if (unlocked) return;
    unlock();
    setTimeout(() => playLoop(), 100);
  };

  // Step progression + sound triggers
  useEffect(() => {
    if (!mounted) return;
    if (hasSeenIntro) {
      onIntroComplete?.();
      window.dispatchEvent(new CustomEvent("grid-break", { detail: { count: 8 } }));
      return;
    }

    if (step === 0) return;

    if (step === 3) {
      playStart();
    }
    if (step === 5) {
      playLoop();
    }

    const ms = STEP_TIMINGS[step];
    if (ms) {
      const timer = setTimeout(() => setStep(step + 1), ms);
      return () => clearTimeout(timer);
    }

    if (step === 6) {
      localStorage.setItem(STORAGE_KEY, "true");
      setHasSeenIntro(true);
      onIntroComplete?.();
      window.dispatchEvent(new CustomEvent("grid-break", { detail: { count: 8 } }));
    }
  }, [step, mounted, hasSeenIntro]);

  // Don't render anything until auth check is done
  if (!mounted || !authChecked) {
    return <section className="h-screen" />;
  }

  const isFirstVisit = !hasSeenIntro;
  const showIntro = isFirstVisit && step >= 1;

  // Intro sequence states
  const showEnter = isFirstVisit && step === 1;
  const showThe = isFirstVisit && step === 3;
  const showTitle = !isFirstVisit || step >= 5;
  const showSubtitle = !isFirstVisit || step >= 6;
  const isPause = isFirstVisit && (step === 2 || step === 4);
  const glitchMode = showTitle
    ? (step === 5 && isFirstVisit ? "reveal" : "ambient")
    : "idle";

  const showLanding = isFirstVisit && step === 0;

  return (
    <>
      {/* ─── Glitch intro overlay ─── */}
      <AnimatePresence>
        {showIntro && (
          <motion.section
            key="intro-overlay"
            initial={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black"
          >
            <div className="flex flex-col items-center">
              <h1 className={TITLE_CLASSES}>
                {showEnter && (
                  <GlitchText text="Enter" mode="fullglitch" />
                )}
                {showThe && (
                  <GlitchText text="The" mode="fullglitch" />
                )}
                {showTitle && (
                  <>
                    <GlitchText
                      text="Crypto"
                      mode={glitchMode}
                      onRevealComplete={() => setStep(6)}
                    />
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.3, delay: 0.5 }}
                    >
                      sensus
                    </motion.span>
                  </>
                )}
                {isPause && <span className="invisible">Cryptosensus</span>}
              </h1>

              {showSubtitle && (
                <div className="mt-8 flex flex-col items-center gap-2.5 text-center">
                  <motion.p
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 1.8, delay: 0.6 }}
                    className="text-base sm:text-lg md:text-xl text-white/40 tracking-wide font-light"
                  >
                    The first{" "}
                    <span className="text-white font-normal">real-time crypto</span>
                    {" "}lowcap{" "}
                    <span className="text-white font-normal">buy score</span>
                  </motion.p>
                  <motion.p
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 1.8, delay: 1.2 }}
                    className="text-sm sm:text-base text-white/25 tracking-wide font-light"
                  >
                    Get high rewards with minimal risk
                  </motion.p>
                </div>
              )}
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      {/* ─── Returning visitor hero (post-auth) ─── */}
      {!isFirstVisit && (
        <section
          className="relative h-screen flex items-center justify-center overflow-hidden"
          onClick={handleUnlockLoop}
        >
          <div className="flex flex-col items-center">
            <h1 className={TITLE_CLASSES}>
              <GlitchText text="Crypto" mode="ambient" />
              <span>sensus</span>
            </h1>
            <div className="mt-8 flex flex-col items-center gap-2.5 text-center">
              <p className="text-base sm:text-lg md:text-xl text-white/40 tracking-wide font-light">
                The first{" "}
                <span className="text-white font-normal">real-time crypto</span>
                {" "}lowcap{" "}
                <span className="text-white font-normal">buy score</span>
              </p>
              <p className="text-sm sm:text-base text-white/25 tracking-wide font-light">
                Get high rewards with minimal risk
              </p>
            </div>
          </div>
        </section>
      )}

      {/* ─── Landing page (not authenticated) ─── */}
      {showLanding && (
        <div
          className="flex flex-col overflow-x-hidden"
          onMouseMove={handleLandingMouseMove}
          onMouseLeave={() => setCursorVisible(false)}
          onClick={handleLandingClick}
        >
          {/* Cursor sound tooltip */}
          {!unlocked && cursorVisible && (
            <div
              className="fixed z-50 pointer-events-none font-mono text-[10px] tracking-wider text-white/40"
              style={{
                left: cursorPos.x + 16,
                top: cursorPos.y + 16,
              }}
            >
              click for sound
            </div>
          )}

          {/* ── Section 1: Hero + Password ── */}
          <section className="min-h-screen flex flex-col items-center justify-center px-6 py-24">
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight text-white text-center leading-[1.1]"
            >
              Join the most advanced
              <br />
              <span className="text-white/70">crypto buy score</span> prediction
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.15 }}
              className="mt-6 text-white/40 text-base sm:text-lg text-center max-w-2xl leading-relaxed"
            >
              Stop wasting your time trading. Know what to buy before the crowd does.
              <br />
              <span className="text-white/25 text-sm sm:text-base">Become an insider</span>
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.3 }}
              className="mt-12 flex flex-col items-center gap-3"
            >
              <span className="text-xs tracking-[0.3em] uppercase text-white/50">
                <GlitchText text="Early Access" mode="ambient" subtle className="text-xs tracking-[0.3em] font-mono" />
              </span>
              <PasswordGate onSuccess={handlePasswordSuccess} inline />
              <button
                onClick={() => {
                  const el = document.getElementById("contact-section");
                  if (el) window.scrollTo({ top: el.offsetTop + 110, behavior: "smooth" });
                }}
                className="text-xs text-white/30 hover:text-white/60 transition-colors cursor-pointer"
              >
                Contact me for access
              </button>
              {/* Sentinel 1: right after "Contact me for access" */}
              <div
                ref={(el) => { sentinelRefs.current[0] = el; }}
                data-sentinel="sentinel-1"
                className="h-px w-full"
                aria-hidden="true"
              />
            </motion.div>

            {/* Features grid */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.5 }}
              className="mt-20 grid grid-cols-2 md:grid-cols-3 gap-4 sm:gap-6 max-w-3xl w-full"
            >
              {FEATURES.map((feature, i) => (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.6 + i * 0.08 }}
                  className="rounded-lg px-4 py-4 sm:px-5 sm:py-5"
                  style={{
                    backgroundColor: "rgba(17, 17, 17, 0.6)",
                    backdropFilter: "blur(12px)",
                    border: "1px solid rgba(255, 255, 255, 0.06)",
                  }}
                >
                  <p className="text-sm font-medium text-white/80">{feature.title}</p>
                  <p className="mt-1 text-xs text-white/35 leading-relaxed">{feature.desc}</p>
                </motion.div>
              ))}
            </motion.div>
          </section>

          {/* Sentinel 2: between features and tagline */}
          <div
            ref={(el) => { sentinelRefs.current[1] = el; }}
            data-sentinel="sentinel-2"
            className="h-px w-full"
            aria-hidden="true"
          />

          {/* ── Section 2: Tagline ── */}
          <section className="py-20 sm:py-28 px-6 flex flex-col items-center">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 0.7 }}
              className="text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight text-center uppercase"
            >
              <span className="text-white/60">Get </span>
              <span style={{ color: "#00ff41" }}>High Rewards</span>
              <span className="text-white/60"> With </span>
              <span style={{ color: "#00ff41" }}>Minimal Risk</span>
            </motion.h2>

            <motion.p
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ duration: 0.7, delay: 0.12 }}
              className="mt-6 text-white/35 text-sm sm:text-base text-center max-w-xl leading-relaxed"
            >
              Real-time AI-powered analysis
            </motion.p>
            <motion.p
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ duration: 0.7, delay: 0.2 }}
              className="mt-2 text-white/25 text-sm sm:text-base text-center max-w-xl leading-relaxed"
            >
              Predict which lowcaps will give you Xs before the crowd buys
            </motion.p>
          </section>

          {/* Sentinel 3: before preview grid */}
          <div
            ref={(el) => { sentinelRefs.current[2] = el; }}
            data-sentinel="sentinel-3"
            className="h-px w-full"
            aria-hidden="true"
          />

          {/* ── Section 3: Preview Grid ── */}
          <section className="py-16 sm:py-24 px-4 sm:px-6 flex flex-col items-center">
            <motion.p
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ duration: 0.6 }}
              className="text-xs font-mono tracking-[0.3em] uppercase text-white/30 mb-10"
            >
              Live Preview
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.8 }}
              className="relative w-full max-w-5xl"
            >
              {/* Fade overlay at bottom */}
              <div
                className="absolute bottom-0 left-0 right-0 h-32 z-10 pointer-events-none"
                style={{ background: "linear-gradient(to top, #000 0%, transparent 100%)" }}
              />

              <div className="perspective-container">
                <div className="grid-3d grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                  {PREVIEW_TOKENS.map((token, i) => {
                    const podium = PODIUM_COLORS[token.rank];

                    return (
                      <motion.div
                        key={token.symbol}
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.5, delay: i * 0.06 }}
                        className="relative flex flex-col justify-between h-[180px] sm:h-[200px] p-5 rounded-2xl border"
                        style={{
                          borderColor: podium?.border ?? "rgba(34,34,34,1)",
                          backgroundColor: podium
                            ? "rgba(17,17,17,0.8)"
                            : "rgba(17,17,17,0.6)",
                        }}
                      >
                        {/* Rank badge for top 3 */}
                        {podium && (
                          <span
                            className="absolute top-3 right-3 text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded-full border"
                            style={{
                              backgroundColor: podium.badgeBg,
                              borderColor: podium.border,
                              color: podium.badge,
                            }}
                          >
                            #{token.rank}
                          </span>
                        )}

                        <span className="text-xl sm:text-2xl font-bold text-white truncate">
                          {token.symbol}
                        </span>

                        <div className="flex-1 flex items-center justify-center">
                          <span
                            className="text-4xl sm:text-5xl font-bold tabular-nums"
                            style={{ color: podium?.score ?? "#ffffff" }}
                          >
                            {token.score}
                          </span>
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          </section>

          {/* Sentinel 4: before contact section */}
          <div
            ref={(el) => { sentinelRefs.current[3] = el; }}
            data-sentinel="sentinel-4"
            className="h-px w-full"
            aria-hidden="true"
          />

          {/* ── Section 4: Contact ── */}
          <section id="contact-section" className="pt-24 pb-10 sm:pt-32 sm:pb-14 px-6">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ duration: 0.7 }}
              className="max-w-2xl mx-auto"
            >
              {/* Section title */}
              <div className="flex items-center gap-4 mb-12">
                <div className="h-px flex-1" style={{ backgroundColor: "rgba(255,255,255,0.06)" }} />
                <span className="text-xs font-mono tracking-[0.3em] uppercase text-white/30">
                  Get In Touch
                </span>
                <div className="h-px flex-1" style={{ backgroundColor: "rgba(255,255,255,0.06)" }} />
              </div>

              <div
                className="rounded-2xl p-8 sm:p-10"
                style={{
                  backgroundColor: "rgba(17, 17, 17, 0.6)",
                  backdropFilter: "blur(24px)",
                  border: "1px solid rgba(255, 255, 255, 0.06)",
                }}
              >
                <div className="text-center">
                  <h3 className="text-2xl sm:text-3xl font-bold text-white mb-3">
                    Want early access?
                  </h3>
                  <p className="text-white/40 text-sm sm:text-base mb-10 leading-relaxed">
                    Cryptosensus is currently invite-only. Reach out to me.
                  </p>
                </div>

                {/* Contact links */}
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <a
                    href="https://x.com/Maximus0Primus"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-center gap-3 px-6 py-3.5 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 transition-colors group"
                  >
                    <span className="text-lg">𝕏</span>
                    <span className="font-mono text-sm text-white/70 group-hover:text-white transition-colors">
                      @S£igneur
                    </span>
                  </a>

                  <a
                    href="https://t.me/Maximus0Primus"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-center gap-3 px-6 py-3.5 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 transition-colors group"
                  >
                    <svg viewBox="0 0 24 24" fill="currentColor" className="size-4 text-white/60 group-hover:text-white transition-colors" role="img" aria-label="Telegram">
                      <path d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.479.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z" />
                    </svg>
                    <span className="font-mono text-sm text-white/70 group-hover:text-white transition-colors">
                      @S£igneur
                    </span>
                  </a>
                </div>

                {/* Or send a message */}
                <div className="mt-8 pt-8" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                  <p className="text-xs font-mono tracking-wider text-white/25 uppercase mb-4 text-center">
                    Or send a message
                  </p>
                  <LandingContactForm />
                </div>
              </div>
            </motion.div>
          </section>

          {/* Sentinel 5: after contact section (densest) */}
          <div
            ref={(el) => { sentinelRefs.current[4] = el; }}
            data-sentinel="sentinel-5"
            className="h-px w-full"
            aria-hidden="true"
          />

          {/* ── Footer spacer ── */}
          <div className="h-6" />
        </div>
      )}
    </>
  );
}

// ─── Inline contact form for landing page ───
function LandingContactForm() {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [status, setStatus] = useState<"idle" | "sending" | "sent" | "error">("idle");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("sending");
    try {
      const res = await fetch("/api/contact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, message }),
      });
      if (!res.ok) throw new Error();
      setStatus("sent");
      setEmail("");
      setMessage("");
    } catch {
      setStatus("error");
    }
  }

  if (status === "sent") {
    return (
      <motion.p
        initial={{ opacity: 0, y: 5 }}
        animate={{ opacity: 1, y: 0 }}
        className="font-mono text-sm text-[#00ff41]/70 py-4"
      >
        Message sent. I&apos;ll get back to you.
      </motion.p>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3">
      <input
        type="email"
        placeholder="your@email.com (optional)"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        className="w-full rounded-lg border border-white/8 bg-white/5 px-4 py-2.5 font-mono text-sm text-white/90 placeholder:text-white/20 outline-none focus:border-white/20 transition-colors"
      />
      <textarea
        required
        placeholder="Your message..."
        rows={3}
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        className="w-full resize-none rounded-lg border border-white/8 bg-white/5 px-4 py-2.5 font-mono text-sm text-white/90 placeholder:text-white/20 outline-none focus:border-white/20 transition-colors"
      />
      {status === "error" && (
        <p className="font-mono text-xs text-red-400">Failed to send. Try again.</p>
      )}
      <button
        type="submit"
        disabled={status === "sending"}
        className="self-start rounded-lg border border-white/10 bg-white/8 px-6 py-2.5 font-mono text-xs tracking-wider text-white/70 uppercase hover:bg-white/15 hover:text-white transition-colors disabled:opacity-40"
      >
        {status === "sending" ? "Sending..." : "Send Message"}
      </button>
    </form>
  );
}
