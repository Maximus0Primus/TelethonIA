"use client";

import { useState, useEffect } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { GlitchText } from "./GlitchText";
import { useAudio } from "@/components/AudioProvider";

const STORAGE_KEY = "cryptosensus_intro_seen_v4";

const TITLE_CLASSES =
  "text-6xl sm:text-7xl md:text-8xl lg:text-9xl font-bold tracking-tight text-white select-none whitespace-nowrap";

// step 0: click gate (wait for user gesture to unlock audio)
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

interface HeroSectionProps {
  onIntroComplete?: () => void;
}

export function HeroSection({ onIntroComplete }: HeroSectionProps) {
  const [mounted, setMounted] = useState(false);
  const [hasSeenIntro, setHasSeenIntro] = useState(false);
  const [step, setStep] = useState(0);

  const { playStart, playLoop, unlock, unlocked } = useAudio();

  const { scrollY } = useScroll();
  const opacity = useTransform(scrollY, [0, 400], [1, 0]);
  const scale = useTransform(scrollY, [0, 400], [1, 0.92]);

  // Mount + localStorage check
  useEffect(() => {
    setMounted(true);
    const seen = localStorage.getItem(STORAGE_KEY);
    if (seen) {
      setHasSeenIntro(true);
    }
  }, []);

  // Click to start intro — user gesture unlocks audio playback
  const handleStart = () => {
    if (unlocked) return;
    unlock();
    playStart();
    setStep(1);
  };

  // Returning visitor: click anywhere to start ambient loop
  const handleUnlockLoop = () => {
    if (unlocked) return;
    unlock();
    // Small delay to let the warm-up finish
    setTimeout(() => playLoop(), 100);
  };

  // Step progression + sound triggers
  useEffect(() => {
    if (!mounted) return;
    if (hasSeenIntro) {
      onIntroComplete?.();
      return;
    }

    // step 0: waiting for click (handleStart)
    if (step === 0) return;

    // step 3: replay start sound
    if (step === 3) {
      playStart();
    }
    // step 5: start ambient loop
    if (step === 5) {
      playLoop();
    }

    const ms = STEP_TIMINGS[step];
    if (ms) {
      const timer = setTimeout(() => setStep(step + 1), ms);
      return () => clearTimeout(timer);
    }

    // step 6 = final
    if (step === 6) {
      localStorage.setItem(STORAGE_KEY, "true");
      onIntroComplete?.();
    }
  }, [step, mounted, hasSeenIntro]);

  if (!mounted) {
    return <section className="h-screen" />;
  }

  const isFirstVisit = !hasSeenIntro;

  // Determine what to show at each step
  const showClickGate = isFirstVisit && step === 0;
  const showEnter = isFirstVisit && step === 1;
  const showThe = isFirstVisit && step === 3;
  const showTitle = !isFirstVisit || step >= 5;
  const showSubtitle = !isFirstVisit || step >= 6;
  const isPause = isFirstVisit && (step === 2 || step === 4);
  const glitchMode = showTitle
    ? (step === 5 && isFirstVisit ? "reveal" : "ambient")
    : "idle";

  return (
    <section
      className="relative h-screen flex items-center justify-center overflow-hidden"
      onClick={hasSeenIntro ? handleUnlockLoop : undefined}
    >
      {showClickGate && (
        <motion.button
          onClick={handleStart}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="absolute inset-0 z-10 flex items-center justify-center cursor-pointer bg-transparent border-none outline-none"
        >
          <motion.span
            animate={{ opacity: [0.3, 0.8, 0.3] }}
            transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
            className="text-xs sm:text-sm text-white/50 tracking-[0.3em] uppercase font-light"
          >
            Click to enter
          </motion.span>
        </motion.button>
      )}

      <motion.div
        style={{ opacity, scale }}
        className="flex flex-col items-center"
      >
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
              transition={{ duration: 1.8, delay: isFirstVisit ? 0.6 : 0 }}
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
              transition={{ duration: 1.8, delay: isFirstVisit ? 1.2 : 0.1 }}
              className="text-sm sm:text-base text-white/25 tracking-wide font-light"
            >
              Get high rewards with minimal risk
            </motion.p>
          </div>
        )}
      </motion.div>
    </section>
  );
}
