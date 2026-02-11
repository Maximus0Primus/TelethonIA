"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion } from "framer-motion";

interface PasswordGateProps {
  onSuccess: (token: string) => void;
  inline?: boolean;
}

export function PasswordGate({ onSuccess, inline = false }: PasswordGateProps) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [shake, setShake] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [blocked, setBlocked] = useState(false);
  const [retryAfter, setRetryAfter] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Auto-focus on mount
  useEffect(() => {
    setTimeout(() => inputRef.current?.focus(), 100);
  }, []);

  // Countdown for blocked state
  useEffect(() => {
    if (!blocked || retryAfter <= 0) return;
    countdownRef.current = setInterval(() => {
      setRetryAfter((prev) => {
        if (prev <= 1) {
          setBlocked(false);
          setError(null);
          if (countdownRef.current) clearInterval(countdownRef.current);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => {
      if (countdownRef.current) clearInterval(countdownRef.current);
    };
  }, [blocked, retryAfter]);

  const formatCountdown = useCallback((seconds: number): string => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }, []);

  const triggerShake = useCallback(() => {
    setShake(true);
    setTimeout(() => setShake(false), 400);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (submitting || blocked || !password.trim()) return;

    setSubmitting(true);
    setError(null);

    try {
      const res = await fetch("/api/auth/verify-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password: password.trim() }),
      });
      const data = await res.json();

      if (res.ok && data.token) {
        onSuccess(data.token);
        return;
      }
      if (res.status === 429 && data.blocked) {
        setBlocked(true);
        setRetryAfter(data.retryAfter || 600);
        setError(null);
        triggerShake();
      } else if (res.status === 401) {
        const remaining = data.attemptsRemaining ?? "?";
        setError(`you can do better — ${remaining} left`);
        triggerShake();
      } else {
        setError("server error");
      }
    } catch {
      setError("network error");
    } finally {
      setSubmitting(false);
      setPassword("");
    }
  };

  const formContent = (
    <motion.div
      className="flex flex-col items-center"
      animate={shake ? { x: [-8, 8, -6, 6, -3, 3, 0] } : { x: 0 }}
      transition={shake ? { duration: 0.4 } : {}}
    >
      <motion.form
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4 }}
        onSubmit={handleSubmit}
        className="flex flex-col items-center gap-3"
      >
        <input
          ref={inputRef}
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          disabled={blocked || submitting}
          autoComplete="off"
          className="w-56 px-4 py-2.5 text-center font-mono text-base tracking-[0.2em] text-white outline-none transition-colors"
          style={{
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            border: "1px solid rgba(255, 255, 255, 0.15)",
            borderRadius: "2px",
          }}
          placeholder={blocked ? "" : "........"}
        />

        {blocked ? (
          <motion.p
            animate={{ opacity: [0.3, 0.6, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
            className="text-xs font-mono tracking-wider text-white/40"
          >
            locked {formatCountdown(retryAfter)}
          </motion.p>
        ) : error ? (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-xs font-mono tracking-wider text-white/40"
          >
            {error}
          </motion.p>
        ) : null}
      </motion.form>
    </motion.div>
  );

  if (inline) {
    return formContent;
  }

  return (
    <div className="fixed inset-0 z-10 flex items-center justify-center pointer-events-none">
      <div className="pointer-events-auto">
        {formContent}
      </div>
    </div>
  );
}
