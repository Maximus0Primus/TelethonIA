"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion } from "framer-motion";

const CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&*!?<>{}[]";
const SCRAMBLE_FPS = 40;
const LOCK_DURATION = 1800;

// Ambient config
const AMBIENT_FLICKER_MIN = 800;
const AMBIENT_FLICKER_MAX = 2000;
const AMBIENT_FLICKER_DURATION = 200;

interface GlitchTextProps {
  text: string;
  mode: "reveal" | "ambient" | "fullglitch" | "idle";
  onRevealComplete?: () => void;
  className?: string;
  subtle?: boolean;
}

export function GlitchText({ text, mode, onRevealComplete, className = "", subtle = false }: GlitchTextProps) {
  const [mounted, setMounted] = useState(false);
  const [displayChars, setDisplayChars] = useState<string[]>(text.split(""));
  const [lockedCount, setLockedCount] = useState(text.length);
  const [flickerIndices, setFlickerIndices] = useState<Set<number>>(new Set());
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lockIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const ambientTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const completedRef = useRef(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const cleanup = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (lockIntervalRef.current) {
      clearInterval(lockIntervalRef.current);
      lockIntervalRef.current = null;
    }
    if (ambientTimerRef.current) {
      clearTimeout(ambientTimerRef.current);
      ambientTimerRef.current = null;
    }
  }, []);

  // Reveal mode: scramble then lock left-to-right
  useEffect(() => {
    if (!mounted || mode !== "reveal") return;

    completedRef.current = false;
    setLockedCount(0);
    setFlickerIndices(new Set());

    intervalRef.current = setInterval(() => {
      setLockedCount((currentLocked) => {
        setDisplayChars(() => {
          const chars = text.split("");
          return chars.map((original, i) => {
            if (i < currentLocked) return original;
            return CHARS[Math.floor(Math.random() * CHARS.length)];
          });
        });
        return currentLocked;
      });
    }, SCRAMBLE_FPS);

    const lockStep = LOCK_DURATION / text.length;
    lockIntervalRef.current = setInterval(() => {
      setLockedCount((prev) => {
        const next = prev + 1;
        if (next >= text.length) {
          if (intervalRef.current) clearInterval(intervalRef.current);
          intervalRef.current = null;
          if (lockIntervalRef.current) clearInterval(lockIntervalRef.current);
          lockIntervalRef.current = null;
          setDisplayChars(text.split(""));
          if (!completedRef.current) {
            completedRef.current = true;
            // Defer to avoid setState-during-render warning
            // (this runs inside a setLockedCount updater)
            queueMicrotask(() => onRevealComplete?.());
          }
        }
        return next;
      });
    }, lockStep);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = null;
      if (lockIntervalRef.current) clearInterval(lockIntervalRef.current);
      lockIntervalRef.current = null;
    };
  }, [mounted, mode, text, onRevealComplete]);

  // Ambient mode: subtle random flickers
  useEffect(() => {
    if (!mounted || mode !== "ambient") {
      setFlickerIndices(new Set());
      if (mode !== "reveal" && mode !== "fullglitch") {
        setDisplayChars(text.split(""));
        setLockedCount(text.length);
      }
      return;
    }

    setDisplayChars(text.split(""));
    setLockedCount(text.length);

    const scheduleFlicker = () => {
      const delay = AMBIENT_FLICKER_MIN + Math.random() * (AMBIENT_FLICKER_MAX - AMBIENT_FLICKER_MIN);
      ambientTimerRef.current = setTimeout(() => {
        const count = Math.random() > 0.6 ? 3 : 2;
        const indices = new Set<number>();
        for (let i = 0; i < count; i++) {
          indices.add(Math.floor(Math.random() * text.length));
        }

        setFlickerIndices(indices);
        setDisplayChars(() => {
          const chars = text.split("");
          indices.forEach((idx) => {
            chars[idx] = CHARS[Math.floor(Math.random() * CHARS.length)];
          });
          return chars;
        });

        setTimeout(() => {
          setFlickerIndices(new Set());
          setDisplayChars(text.split(""));
        }, AMBIENT_FLICKER_DURATION);

        scheduleFlicker();
      }, delay);
    };

    scheduleFlicker();

    return () => {
      if (ambientTimerRef.current) {
        clearTimeout(ambientTimerRef.current);
        ambientTimerRef.current = null;
      }
    };
  }, [mounted, mode, text]);

  // Fullglitch mode: all chars styled green, rapid random swaps
  useEffect(() => {
    if (!mounted || mode !== "fullglitch") return;

    setDisplayChars(text.split(""));
    setLockedCount(text.length);

    const flickerLoop = setInterval(() => {
      const count = 1 + Math.floor(Math.random() * 2); // 1-2 chars
      const indices = new Set<number>();
      for (let i = 0; i < count; i++) {
        indices.add(Math.floor(Math.random() * text.length));
      }
      setFlickerIndices(indices);
      setDisplayChars(() => {
        const chars = text.split("");
        indices.forEach((idx) => {
          chars[idx] = CHARS[Math.floor(Math.random() * CHARS.length)];
        });
        return chars;
      });

      setTimeout(() => {
        setFlickerIndices(new Set());
        setDisplayChars(text.split(""));
      }, 50);
    }, 120);

    return () => clearInterval(flickerLoop);
  }, [mounted, mode, text]);

  // Cleanup on unmount
  useEffect(() => cleanup, [cleanup]);

  if (!mounted) {
    return <span className={className}>{text}</span>;
  }

  const isRevealing = mode === "reveal";
  const isAmbient = mode === "ambient";
  const isFullGlitch = mode === "fullglitch";

  return (
    <motion.span
      className={`inline-flex ${className}`}
      animate={
        isFullGlitch
          ? { x: [0, -3, 5, -4, 0] }
          : isRevealing
            ? { x: [0, -1, 2, -1, 0] }
            : isAmbient
              ? subtle
                ? { x: [0, 0.3, -0.2, 0.3, 0] }
                : { x: [0, 1.5, -1.2, 1.8, 0] }
              : { x: 0 }
      }
      transition={
        isFullGlitch
          ? { duration: 0.15, repeat: Infinity, repeatType: "mirror" }
          : isRevealing
            ? { duration: 0.3, repeat: Infinity, repeatType: "mirror" }
            : isAmbient
              ? { duration: 2, repeat: Infinity, repeatType: "mirror", ease: "easeInOut" }
              : {}
      }
    >
      {displayChars.map((char, i) => {
        const isLocked = i < lockedCount;
        const isFlickering = flickerIndices.has(i);

        const showGlitchStyle = (isRevealing && !isLocked) || isFlickering || isFullGlitch;

        return (
          <motion.span
            key={i}
            className="inline-block"
            style={{
              fontFamily: showGlitchStyle ? "var(--font-jetbrains-mono), monospace" : "inherit",
              color: showGlitchStyle ? "#00ff41" : "inherit",
              textShadow: showGlitchStyle
                ? subtle
                  ? "0 0 6px rgba(0,255,65,0.4)"
                  : "3px 0 #ff0040, -3px 0 #00ff41"
                : isAmbient
                  ? subtle
                    ? "0 0 4px rgba(0,255,65,0.2)"
                    : "2px 0 rgba(255,0,64,0.6), -2px 0 rgba(0,255,65,0.6)"
                  : "none",
              minWidth: "0.6em",
              textAlign: "center",
            }}
          >
            {char}
          </motion.span>
        );
      })}
    </motion.span>
  );
}
