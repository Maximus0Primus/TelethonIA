"use client";

import { createContext, useContext, useRef, useCallback, useEffect, useState } from "react";

interface AudioContextValue {
  /** Play the one-shot start sound */
  playStart: () => void;
  /** Start the ambient loop (requires prior user gesture) */
  playLoop: () => void;
  /** Whether audio has been unlocked by a user gesture */
  unlocked: boolean;
  /** Unlock audio in a click handler — must be called from a user gesture */
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
    // Only play if not already playing
    if (loopRef.current.paused) {
      loopRef.current.play().catch(() => {});
    }
  }, []);

  return (
    <AudioCtx.Provider value={{ playStart, playLoop, unlocked, unlock }}>
      {children}
    </AudioCtx.Provider>
  );
}
