"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { useAudio } from "@/components/AudioProvider";

export function Header() {
  const { toggleLoop, loopPlaying } = useAudio();

  return (
    <motion.header
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
      className="fixed top-0 left-0 right-0 z-50 pointer-events-none"
    >
      <div className="mx-auto flex h-20 max-w-7xl items-center justify-between px-6 sm:px-8">
        <Link
          href="/"
          className="pointer-events-auto"
        >
          <h1 className="text-xl font-bold tracking-tight text-white hover:text-white/80 transition-colors">
            Cryptosensus
          </h1>
        </Link>

        <button
          onClick={toggleLoop}
          className="pointer-events-auto text-[11px] font-mono tracking-wider text-white/40 hover:text-white/70 transition-colors uppercase"
        >
          sound [{loopPlaying ? "on" : "off"}]
        </button>
      </div>
    </motion.header>
  );
}
