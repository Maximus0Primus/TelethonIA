"use client";

import { useEffect, useRef, type MutableRefObject } from "react";

interface MousePos {
  x: number;
  y: number;
}

const OFF_SCREEN: MousePos = { x: -1000, y: -1000 };

export function useMousePosition(): MutableRefObject<MousePos> {
  const ref = useRef<MousePos>({ ...OFF_SCREEN });

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      ref.current.x = e.clientX;
      ref.current.y = e.clientY;
    };
    const onLeave = () => {
      ref.current.x = OFF_SCREEN.x;
      ref.current.y = OFF_SCREEN.y;
    };

    window.addEventListener("mousemove", onMove, { passive: true });
    document.addEventListener("mouseleave", onLeave);

    return () => {
      window.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseleave", onLeave);
    };
  }, []);

  return ref;
}
