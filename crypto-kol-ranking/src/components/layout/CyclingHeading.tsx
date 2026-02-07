"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const CATEGORIES = ["MEMECOIN", "AI COIN", "DEFI COIN", "TECH COIN"];
const CYCLE_INTERVAL = 1800;

const slideUp = {
  initial: { y: "100%", opacity: 0 },
  animate: { y: "0%", opacity: 1 },
  exit: { y: "-100%", opacity: 0 },
};

const transition = {
  duration: 0.3,
  ease: [0.16, 1, 0.3, 1] as const,
};

export function CyclingHeading() {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % CATEGORIES.length);
    }, CYCLE_INTERVAL);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center justify-center py-10 sm:py-14">
      <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight text-white flex items-center gap-0">
        {/* Container shrinks/grows to fit the current word via layout animation */}
        <motion.span
          className="relative inline-flex overflow-hidden items-center"
          layout
          transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
        >
          {/* Invisible sizer: only the CURRENT word, sets natural width */}
          <span className="invisible whitespace-nowrap" aria-hidden="true">
            {CATEGORIES[index]}
          </span>
          <AnimatePresence mode="popLayout">
            <motion.span
              key={CATEGORIES[index]}
              variants={slideUp}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={transition}
              className="absolute left-0 top-0 inline-block whitespace-nowrap"
              style={{ color: "#00ff41" }}
            >
              {CATEGORIES[index]}
            </motion.span>
          </AnimatePresence>
        </motion.span>
        <span className="text-white/60">&nbsp;TO BUY NOW</span>
      </h2>
    </div>
  );
}
