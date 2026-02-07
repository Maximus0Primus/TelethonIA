"use client";

import { useEffect, useState } from "react";

interface ParallaxOptions {
  speed?: number;
  disabled?: boolean;
}

export function useParallax({ speed = 0.1, disabled = false }: ParallaxOptions = {}) {
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (disabled) return;

    let rafId: number;
    let currentY = 0;

    const handleScroll = () => {
      rafId = requestAnimationFrame(() => {
        currentY = window.scrollY;
        setOffset({
          x: 0,
          y: currentY * speed,
        });
      });
    };

    window.addEventListener("scroll", handleScroll, { passive: true });

    return () => {
      window.removeEventListener("scroll", handleScroll);
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [speed, disabled]);

  return offset;
}

// Hook for detecting mobile/touch devices
export function useIsMobile() {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.matchMedia("(max-width: 768px)").matches);
    };

    checkMobile();
    window.addEventListener("resize", checkMobile);

    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  return isMobile;
}
