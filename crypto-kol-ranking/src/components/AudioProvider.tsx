"use client";

import { createContext, useContext, useRef, useCallback, useEffect, useState } from "react";

interface AudioContextValue {
  playStart: () => void;
  playLoop: () => void;
  stopLoop: () => void;
  toggleLoop: () => void;
  loopPlaying: boolean;
  unlocked: boolean;
  unlock: () => void;
}

const AudioCtx = createContext<AudioContextValue>({
  playStart: () => {},
  playLoop: () => {},
  stopLoop: () => {},
  toggleLoop: () => {},
  loopPlaying: false,
  unlocked: false,
  unlock: () => {},
});

export const useAudio = () => useContext(AudioCtx);

export function AudioProvider({ children }: { children: React.ReactNode }) {
  const startRef = useRef<HTMLAudioElement | null>(null);
  const loopRef = useRef<HTMLAudioElement | null>(null);
  const unlockedRef = useRef(false);
  const [unlocked, setUnlocked] = useState(false);
  const [loopPlaying, setLoopPlaying] = useState(false);

  // Create audio elements once, persist across navigations
  useEffect(() => {
    startRef.current = new Audio("/audio/start.mp3");
    startRef.current.preload = "auto";
    loopRef.current = new Audio("/audio/loop.mp3");
    loopRef.current.loop = true;
    loopRef.current.preload = "auto";

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
      loopRef.current
        .play()
        .then(() => setLoopPlaying(true))
        .catch(() => setLoopPlaying(false));
    }
  }, []);

  const stopLoop = useCallback(() => {
    if (!loopRef.current) return;
    loopRef.current.pause();
    setLoopPlaying(false);
  }, []);

  const toggleLoop = useCallback(() => {
    if (!loopRef.current) return;

    // First toggle also unlocks audio
    if (!unlockedRef.current) {
      unlockedRef.current = true;
      setUnlocked(true);
    }

    if (loopRef.current.paused) {
      loopRef.current
        .play()
        .then(() => setLoopPlaying(true))
        .catch(() => setLoopPlaying(false));
    } else {
      loopRef.current.pause();
      setLoopPlaying(false);
    }
  }, []);

  return (
    <AudioCtx.Provider value={{ playStart, playLoop, stopLoop, toggleLoop, loopPlaying, unlocked, unlock }}>
      {children}
    </AudioCtx.Provider>
  );
}
