"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { useAudio } from "@/components/AudioProvider";
import { ContactModal } from "@/components/layout/ContactModal";

export function Header() {
  const { toggleLoop, loopPlaying } = useAudio();
  const [modalOpen, setModalOpen] = useState(false);

  const handleLetsTalk = () => {
    const el = document.getElementById("contact-section");
    if (el) {
      window.scrollTo({ top: el.offsetTop + 110, behavior: "smooth" });
    } else {
      setModalOpen(true);
    }
  };

  return (
    <>
      <motion.header
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="fixed top-0 left-0 right-0 z-50 pointer-events-none"
      >
        <div className="mx-auto flex h-20 max-w-7xl items-center justify-between px-4 sm:px-8">
          <Link
            href="/"
            className="pointer-events-auto"
          >
            <h1 className="text-xl font-bold tracking-tight text-white hover:text-white/80 transition-colors">
              Cryptosensus
            </h1>
          </Link>

          <div className="flex items-center gap-2 sm:gap-4">
            <button
              onClick={toggleLoop}
              className="pointer-events-auto text-[11px] font-mono tracking-wider text-white/40 hover:text-white/70 transition-colors uppercase"
            >
              sound [{loopPlaying ? "on" : "off"}]
            </button>
            <button
              onClick={handleLetsTalk}
              className="pointer-events-auto rounded-full border border-white bg-white px-4 py-1.5 font-mono text-xs tracking-wider text-black hover:bg-white/85 transition-colors uppercase"
            >
              Let&apos;s Talk
            </button>
          </div>
        </div>
      </motion.header>

      <ContactModal open={modalOpen} onOpenChange={setModalOpen} trigger={<span />} />
    </>
  );
}
