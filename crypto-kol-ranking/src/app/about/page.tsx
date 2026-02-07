"use client";

import { motion } from "framer-motion";
import { Header } from "@/components/layout/Header";
import { FloatingNav } from "@/components/layout/FloatingNav";

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-[#F0F0F0]">
      {/* Override header for light background */}
      <motion.header
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="fixed top-0 left-0 right-0 z-50 pointer-events-none"
      >
        <div className="mx-auto flex h-20 max-w-7xl items-center px-6 sm:px-8">
          <a href="/" className="pointer-events-auto">
            <h1 className="text-xl font-bold tracking-tight text-black hover:text-black/70 transition-colors">
              Cryptosensus
            </h1>
          </a>
        </div>
      </motion.header>

      {/* Custom nav for light theme */}
      <motion.nav
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50"
      >
        <div className="flex items-center gap-1 rounded-full px-2 py-2 bg-white/80 backdrop-blur-xl border border-black/10">
          <a
            href="/"
            className="px-5 py-2 text-sm font-medium rounded-full text-black/60 hover:text-black transition-colors"
          >
            Tokens
          </a>
          <span className="relative px-5 py-2 text-sm font-medium rounded-full text-white">
            <span className="absolute inset-0 bg-black rounded-full" />
            <span className="relative z-10">About</span>
          </span>
        </div>
      </motion.nav>

      <main className="pt-32 pb-40 px-6 sm:px-8">
        <div className="max-w-4xl mx-auto">
          {/* Title */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="mb-20"
          >
            <h1 className="text-5xl sm:text-7xl md:text-8xl font-bold tracking-tight text-black">
              Cryptosensus
            </h1>
          </motion.div>

          {/* Manifesto sections */}
          <motion.section
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="manifesto-section pt-12 pb-16 border-t border-black/10"
          >
            <p className="text-2xl sm:text-3xl md:text-4xl leading-relaxed text-black/70">
              Cryptosensus was created to{" "}
              <span className="text-black font-medium">bring clarity to the noise</span>.
            </p>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="manifesto-section pt-12 pb-16 border-t border-black/10"
          >
            <p className="text-2xl sm:text-3xl md:text-4xl leading-relaxed text-black/70">
              We track sentiment from{" "}
              <span className="text-black font-medium">50+ crypto KOLs</span>{" "}
              to surface the tokens that matter.
            </p>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="manifesto-section pt-12 pb-16 border-t border-black/10"
          >
            <p className="text-2xl sm:text-3xl md:text-4xl leading-relaxed text-black/70">
              Not hype. Not shills. Just{" "}
              <span className="text-black font-medium">data-driven insights</span>{" "}
              that help you make informed decisions.
            </p>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="pt-12 pb-16 border-t border-black/10"
          >
            <div className="grid sm:grid-cols-3 gap-8">
              <div>
                <h3 className="text-sm font-medium text-black/40 uppercase tracking-wider mb-3">
                  KOLs Tracked
                </h3>
                <p className="text-4xl sm:text-5xl font-bold text-black">50+</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-black/40 uppercase tracking-wider mb-3">
                  Messages Analyzed
                </h3>
                <p className="text-4xl sm:text-5xl font-bold text-black">24h</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-black/40 uppercase tracking-wider mb-3">
                  Sentiment Methods
                </h3>
                <p className="text-4xl sm:text-5xl font-bold text-black">3</p>
              </div>
            </div>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="pt-12 border-t border-black/10"
          >
            <h3 className="text-sm font-medium text-black/40 uppercase tracking-wider mb-6">
              How It Works
            </h3>
            <div className="space-y-4">
              <p className="text-lg text-black/70">
                <span className="text-black font-medium">1.</span> Scrape messages from top crypto Telegram groups
              </p>
              <p className="text-lg text-black/70">
                <span className="text-black font-medium">2.</span> Extract token mentions and analyze sentiment using VADER, CryptoBERT, and custom lexicon
              </p>
              <p className="text-lg text-black/70">
                <span className="text-black font-medium">3.</span> Weight by KOL conviction score and aggregate into consensus rankings
              </p>
              <p className="text-lg text-black/70">
                <span className="text-black font-medium">4.</span> Display real-time insights in a clean, actionable format
              </p>
            </div>
          </motion.section>
        </div>
      </main>
    </div>
  );
}
