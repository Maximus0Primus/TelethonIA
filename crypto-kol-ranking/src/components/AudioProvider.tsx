"use client";

import { createContext, useContext, useRef, useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "cryptosensus_intro_seen_v4";

interface AudioContextValue {
  playStart: () => void;
  playLoop: () => void;
  unlocked: boolean;
  unlock: () => void;
}

const AudioCtx = createContext<AudioContextValue>({
  playStart: () => {},
  playLoop: () => {},
  unlocked: false,
  unlock: () => {},
});

export const useAudio = () => useContext(AudioCtx);

export function AudioProvider({ children }: { children: React.ReactNode }) {
  const startRef = useRef<HTMLAudioElement | null>(null);
  const loopRef = useRef<HTMLAudioElement | null>(null);
  const unlockedRef = useRef(false);
  const [unlocked, setUnlocked] = useState(false);

  // Create audio elements once, persist across navigations
  useEffect(() => {
    startRef.current = new Audio("/audio/start.wav");
    loopRef.current = new Audio("/audio/loop.wav");
    loopRef.current.loop = true;

    return () => {
      startRef.current?.pause();
      loopRef.current?.pause();
      startRef.current = null;
      loopRef.current = null;
    };
  }, []);

  const unlock = useCallback(() => {
    if (unlockedRef.current) return;
    unlockedRef.current = true;
    setUnlocked(true);

    // Warm up loop in user gesture context
    if (loopRef.current) {
      loopRef.current.volume = 0;
      loopRef.current.play().then(() => {
        loopRef.current!.pause();
        loopRef.current!.currentTime = 0;
        loopRef.current!.volume = 1;
      }).catch(() => {});
    }
  }, []);

  const playStart = useCallback(() => {
    if (!startRef.current) return;
    startRef.current.currentTime = 0;
    startRef.current.play().catch(() => {});
  }, []);

  const playLoop = useCallback(() => {
    if (!loopRef.current) return;
    if (loopRef.current.paused) {
      loopRef.current.play().catch(() => {});
    }
  }, []);

  // Auto-resume loop on first interaction for returning visitors
  useEffect(() => {
    const seen = typeof window !== "undefined" && localStorage.getItem(STORAGE_KEY);
    if (!seen) return;

    const resumeOnInteraction = () => {
      if (unlockedRef.current) return;
      unlockedRef.current = true;
      setUnlocked(true);

      if (loopRef.current) {
        loopRef.current.play().catch(() => {});
      }

      document.removeEventListener("click", resumeOnInteraction);
      document.removeEventListener("touchstart", resumeOnInteraction);
    };

    document.addEventListener("click", resumeOnInteraction, { once: true });
    document.addEventListener("touchstart", resumeOnInteraction, { once: true });

    return () => {
      document.removeEventListener("click", resumeOnInteraction);
      document.removeEventListener("touchstart", resumeOnInteraction);
    };
  }, []);

  return (
    <AudioCtx.Provider value={{ playStart, playLoop, unlocked, unlock }}>
      {children}
    </AudioCtx.Provider>
  );
}
