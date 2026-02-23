"use client";

import { useEffect, useRef, useCallback } from "react";

interface UseAutoRefreshOptions {
  interval?: number; // polling interval in ms (default 60s)
  enabled?: boolean;
  onDataChange: () => void;
}

export function useAutoRefresh({
  interval = 900_000,
  enabled = true,
  onDataChange,
}: UseAutoRefreshOptions) {
  const lastUpdatedAt = useRef<string | null>(null);
  const onDataChangeRef = useRef(onDataChange);
  onDataChangeRef.current = onDataChange;

  const checkForUpdates = useCallback(async () => {
    try {
      const res = await fetch("/api/ranking/updated-at");
      if (!res.ok) return;

      const { updated_at } = await res.json();
      if (!updated_at) return;

      // First poll: just store the value
      if (lastUpdatedAt.current === null) {
        lastUpdatedAt.current = updated_at;
        return;
      }

      // Subsequent polls: compare
      if (updated_at !== lastUpdatedAt.current) {
        lastUpdatedAt.current = updated_at;
        onDataChangeRef.current();
      }
    } catch {
      // Silently ignore polling errors
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;

    // Initial check
    checkForUpdates();

    const id = setInterval(checkForUpdates, interval);
    return () => clearInterval(id);
  }, [enabled, interval, checkForUpdates]);
}
