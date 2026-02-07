"use client";

import { motion, type HTMLMotionProps } from "framer-motion";
import { cn } from "@/lib/utils";

interface GlowButtonProps extends HTMLMotionProps<"button"> {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md" | "lg";
  glow?: boolean;
}

const variantStyles = {
  primary: "bg-blue-600 hover:bg-blue-500 text-white border-blue-500/50",
  secondary: "bg-zinc-800 hover:bg-zinc-700 text-zinc-100 border-zinc-700",
  ghost: "bg-transparent text-foreground border-transparent hover:bg-zinc-800/50",
};

const sizeStyles = {
  sm: "px-3 py-1.5 text-sm",
  md: "px-4 py-2 text-sm",
  lg: "px-6 py-3 text-base",
};

export function GlowButton({
  children,
  className,
  variant = "primary",
  size = "md",
  glow: _glow,
  disabled,
  ...props
}: GlowButtonProps) {
  return (
    <motion.button
      className={cn(
        "relative inline-flex items-center justify-center gap-2 rounded-lg border font-medium transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        "disabled:pointer-events-none disabled:opacity-50",
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
      whileTap={!disabled ? { scale: 0.98 } : undefined}
      disabled={disabled}
      {...props}
    >
      {children}
    </motion.button>
  );
}
