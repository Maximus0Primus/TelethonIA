"use client";

import { useEffect, useRef, useCallback } from "react";
import { useMousePosition } from "@/hooks/useMousePosition";
import { useIsMobile } from "@/hooks/useParallax";

const GRID_SPACING = 60;
const LINE_OPACITY = 0.07;
const EFFECT_RADIUS = 150;
const BASE_STRENGTH = 25;
const LERP_FACTOR = 0.15;

// Click fury system
const CLICK_DECAY = 0.994;
const CLICK_BOOST = 6;
const FURY_MAX = 100;
const BREAK_THRESHOLD = 80;
const BREAK_DURATION = 20_000;

const PERMANENT_FADE_IN = 600; // ms for permanent cells to fade in

interface BrokenCell {
  col: number; // floor-based cell index
  row: number;
  t: number;
  intensity: number;
  permanent?: boolean;
  group?: string;
}

export function GridDistortion() {
  const isMobile = useIsMobile();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useMousePosition();
  const smoothMouse = useRef({ x: -1000, y: -1000 });
  const rafRef = useRef<number>(0);

  const fury = useRef(0);
  const brokenCells = useRef<BrokenCell[]>([]);

  const handleClick = useCallback((e: MouseEvent) => {
    fury.current = Math.min(fury.current + CLICK_BOOST, FURY_MAX);

    if (fury.current >= BREAK_THRESHOLD) {
      const now = Date.now();
      // Cell = the rectangle between grid lines, found via floor
      const col = Math.floor(e.clientX / GRID_SPACING);
      const row = Math.floor(e.clientY / GRID_SPACING);

      const existing = brokenCells.current.find((b) => b.col === col && b.row === row);
      if (existing) {
        existing.t = now;
        existing.intensity = Math.min(1, existing.intensity + 0.15);
      } else {
        const overflow = (fury.current - BREAK_THRESHOLD) / (FURY_MAX - BREAK_THRESHOLD);
        brokenCells.current.push({
          col,
          row,
          t: now,
          intensity: 0.3 + overflow * 0.7,
        });
      }
    }
  }, []);

  const handleGridBreak = useCallback((e: Event) => {
    const { count } = (e as CustomEvent).detail as { count: number };
    const now = Date.now();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const w = canvas.getBoundingClientRect().width;
    const h = canvas.getBoundingClientRect().height;
    const maxCol = Math.floor(w / GRID_SPACING);
    const maxRow = Math.floor(h / GRID_SPACING);

    const used = new Set<string>();
    for (let i = 0; i < count; i++) {
      let col: number, row: number, key: string;
      let attempts = 0;
      do {
        col = Math.floor(Math.random() * maxCol);
        row = Math.floor(Math.random() * maxRow);
        key = `${col},${row}`;
        attempts++;
      } while (used.has(key) && attempts < 50);
      used.add(key);

      brokenCells.current.push({ col, row, t: now, intensity: 0.6 + Math.random() * 0.4 });
    }
  }, []);

  const handleGridBreakPermanent = useCallback((e: Event) => {
    const { count, minIntensity = 0.15, maxIntensity = 0.6, group } = (e as CustomEvent).detail as {
      count: number;
      minIntensity?: number;
      maxIntensity?: number;
      group?: string;
    };
    const canvas = canvasRef.current;
    if (!canvas) return;

    const w = canvas.getBoundingClientRect().width;
    const h = canvas.getBoundingClientRect().height;
    const maxCol = Math.floor(w / GRID_SPACING);
    const maxRow = Math.floor(h / GRID_SPACING);

    // Collect existing permanent keys to avoid duplicates
    const existingKeys = new Set(
      brokenCells.current.filter((b) => b.permanent).map((b) => `${b.col},${b.row}`)
    );

    // Pre-generate all cell positions, then stagger their addition
    const cellsToAdd: { col: number; row: number; intensity: number }[] = [];
    for (let i = 0; i < count; i++) {
      let col: number, row: number, key: string;
      let attempts = 0;
      do {
        col = Math.floor(Math.random() * maxCol);
        row = Math.floor(Math.random() * maxRow);
        key = `${col},${row}`;
        attempts++;
      } while (existingKeys.has(key) && attempts < 50);
      existingKeys.add(key);
      cellsToAdd.push({
        col,
        row,
        intensity: minIntensity + Math.random() * (maxIntensity - minIntensity),
      });
    }

    // Stagger: add 2 cells every 80ms for a fluid appearance
    const BATCH_SIZE = 2;
    const BATCH_DELAY = 80;
    for (let i = 0; i < cellsToAdd.length; i += BATCH_SIZE) {
      const batch = cellsToAdd.slice(i, i + BATCH_SIZE);
      const delay = (i / BATCH_SIZE) * BATCH_DELAY;
      setTimeout(() => {
        const now = Date.now();
        for (const cell of batch) {
          brokenCells.current.push({
            ...cell,
            t: now,
            permanent: true,
            group,
          });
        }
      }, delay);
    }
  }, []);

  const handleGridClearGroup = useCallback((e: Event) => {
    const { group } = (e as CustomEvent).detail as { group: string };
    brokenCells.current = brokenCells.current.filter((b) => b.group !== group);
  }, []);

  // ── Main effect: canvas draw loop ──
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Desktop-only: click fury
    if (!isMobile) {
      window.addEventListener("click", handleClick);
    }
    // Both: grid-break events
    window.addEventListener("grid-break", handleGridBreak);
    window.addEventListener("grid-break-permanent", handleGridBreakPermanent);
    window.addEventListener("grid-clear-group", handleGridClearGroup);

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    resize();

    // Helper: compute broken cell alpha
    const cellAlpha = (cell: BrokenCell, now: number): number => {
      if (cell.permanent) {
        const age = now - cell.t;
        return cell.intensity * Math.min(1, age / PERMANENT_FADE_IN);
      }
      const age = (now - cell.t) / BREAK_DURATION;
      const fade = age < 0.6 ? 1 : 1 - (age - 0.6) / 0.4;
      return fade * cell.intensity;
    };

    const draw = () => {
      const w = canvas.getBoundingClientRect().width;
      const h = canvas.getBoundingClientRect().height;

      ctx.clearRect(0, 0, w, h);

      const now = Date.now();

      // Prune expired non-permanent cells
      brokenCells.current = brokenCells.current.filter(
        (b) => b.permanent || now - b.t < BREAK_DURATION
      );

      const hasBroken = brokenCells.current.length > 0;

      // Desktop-only: fury + mouse
      let furyNorm = 0;
      let strength = BASE_STRENGTH;
      let radius = EFFECT_RADIUS;
      let r2 = radius * radius;
      let sx = -1000;
      let sy = -1000;

      if (!isMobile) {
        fury.current *= CLICK_DECAY;
        if (fury.current < 0.1) fury.current = 0;

        const mx = mouseRef.current.x;
        const my = mouseRef.current.y;
        smoothMouse.current.x += (mx - smoothMouse.current.x) * LERP_FACTOR;
        smoothMouse.current.y += (my - smoothMouse.current.y) * LERP_FACTOR;
        sx = smoothMouse.current.x;
        sy = smoothMouse.current.y;

        furyNorm = fury.current / FURY_MAX;
        const furyMult = 1 + furyNorm * 6;
        strength = BASE_STRENGTH * furyMult;
        radius = EFFECT_RADIUS + fury.current * 2.5;
        r2 = radius * radius;
      }

      // ── Broken cells: green fill ──
      const brokenMap = new Map<string, BrokenCell>();
      for (const cell of brokenCells.current) {
        const alpha = cellAlpha(cell, now);
        const x0 = cell.col * GRID_SPACING;
        const y0 = cell.row * GRID_SPACING;
        ctx.fillStyle = `rgba(34, 197, 94, ${alpha * 0.35})`;
        ctx.fillRect(x0, y0, GRID_SPACING, GRID_SPACING);
        brokenMap.set(`${cell.col},${cell.row}`, cell);
      }

      // ── Grid lines ──
      const lineAlpha = LINE_OPACITY + furyNorm * 0.04;
      ctx.lineWidth = 1;

      const colCount = Math.ceil(w / GRID_SPACING);
      const rowCount = Math.ceil(h / GRID_SPACING);

      const vLineBroken = (ci: number, cellRow: number): BrokenCell | null => {
        return brokenMap.get(`${ci},${cellRow}`) || brokenMap.get(`${ci - 1},${cellRow}`) || null;
      };
      const hLineBroken = (cellCol: number, ri: number): BrokenCell | null => {
        return brokenMap.get(`${cellCol},${ri}`) || brokenMap.get(`${cellCol},${ri - 1}`) || null;
      };

      const getLineColor = (cell: BrokenCell | null, baseAlpha: number): string => {
        if (!cell) return `rgba(255, 255, 255, ${baseAlpha})`;
        const a = cellAlpha(cell, now);
        return `rgba(34, 197, 94, ${Math.max(baseAlpha, a * 0.5)})`;
      };

      // Distortion (desktop only — on mobile these are identity functions)
      const distortX = isMobile
        ? (baseX: number) => baseX
        : (baseX: number, y: number, jitter: boolean): number => {
            let px = baseX;
            const dx = px - sx;
            const dy = y - sy;
            const d2 = dx * dx + dy * dy;
            if (d2 < r2 && d2 > 0) {
              const d = Math.sqrt(d2);
              const t = 1 - d / radius;
              px += (dx / d) * strength * t * t * t * t;
            }
            if (jitter) px += (Math.random() - 0.5) * 4;
            return px;
          };

      const distortY = isMobile
        ? (_x: number, baseY: number) => baseY
        : (x: number, baseY: number, jitter: boolean): number => {
            let py = baseY;
            const dx = x - sx;
            const dy = py - sy;
            const d2 = dx * dx + dy * dy;
            if (d2 < r2 && d2 > 0) {
              const d = Math.sqrt(d2);
              const t = 1 - d / radius;
              py += (dy / d) * strength * t * t * t * t;
            }
            if (jitter) py += (Math.random() - 0.5) * 4;
            return py;
          };

      // Vertical lines
      for (let ci = 0; ci <= colCount; ci++) {
        const baseX = ci * GRID_SPACING;
        ctx.beginPath();
        let prevCellRow = 0;
        let broken = vLineBroken(ci, 0);

        // Mobile: simple straight lines (no per-pixel distortion)
        if (isMobile) {
          if (hasBroken) {
            // Draw per-segment to color broken segments green
            for (let ri = 0; ri <= rowCount; ri++) {
              const y0 = ri * GRID_SPACING;
              const y1 = Math.min((ri + 1) * GRID_SPACING, h);
              const seg = vLineBroken(ci, ri);
              ctx.beginPath();
              ctx.moveTo(baseX, y0);
              ctx.lineTo(baseX, y1);
              ctx.strokeStyle = getLineColor(seg, lineAlpha);
              ctx.stroke();
            }
          } else {
            ctx.beginPath();
            ctx.moveTo(baseX, 0);
            ctx.lineTo(baseX, h);
            ctx.strokeStyle = `rgba(255, 255, 255, ${lineAlpha})`;
            ctx.stroke();
          }
          continue;
        }

        // Desktop: per-pixel distortion
        for (let y = 0; y <= h; y += 4) {
          const cellRow = Math.floor(y / GRID_SPACING);
          if (cellRow !== prevCellRow) {
            ctx.strokeStyle = getLineColor(broken, lineAlpha);
            ctx.stroke();
            ctx.beginPath();
            broken = vLineBroken(ci, cellRow);
            ctx.moveTo(distortX(baseX, y, !!broken), y);
            prevCellRow = cellRow;
            continue;
          }
          const px = distortX(baseX, y, !!broken);
          if (y === 0) ctx.moveTo(px, y);
          else ctx.lineTo(px, y);
        }
        ctx.strokeStyle = getLineColor(broken, lineAlpha);
        ctx.stroke();
      }

      // Horizontal lines
      for (let ri = 0; ri <= rowCount; ri++) {
        const baseY = ri * GRID_SPACING;

        // Mobile: simple straight lines
        if (isMobile) {
          if (hasBroken) {
            for (let ci = 0; ci <= colCount; ci++) {
              const x0 = ci * GRID_SPACING;
              const x1 = Math.min((ci + 1) * GRID_SPACING, w);
              const seg = hLineBroken(ci, ri);
              ctx.beginPath();
              ctx.moveTo(x0, baseY);
              ctx.lineTo(x1, baseY);
              ctx.strokeStyle = getLineColor(seg, lineAlpha);
              ctx.stroke();
            }
          } else {
            ctx.beginPath();
            ctx.moveTo(0, baseY);
            ctx.lineTo(w, baseY);
            ctx.strokeStyle = `rgba(255, 255, 255, ${lineAlpha})`;
            ctx.stroke();
          }
          continue;
        }

        // Desktop: per-pixel distortion
        ctx.beginPath();
        let prevCellCol = 0;
        let broken = hLineBroken(0, ri);

        for (let x = 0; x <= w; x += 4) {
          const cellCol = Math.floor(x / GRID_SPACING);
          if (cellCol !== prevCellCol) {
            ctx.strokeStyle = getLineColor(broken, lineAlpha);
            ctx.stroke();
            ctx.beginPath();
            broken = hLineBroken(cellCol, ri);
            ctx.moveTo(x, distortY(x, baseY, !!broken));
            prevCellCol = cellCol;
            continue;
          }
          const py = distortY(x, baseY, !!broken);
          if (x === 0) ctx.moveTo(x, py);
          else ctx.lineTo(x, py);
        }
        ctx.strokeStyle = getLineColor(broken, lineAlpha);
        ctx.stroke();
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
      if (!isMobile) {
        window.removeEventListener("click", handleClick);
      }
      window.removeEventListener("grid-break", handleGridBreak);
      window.removeEventListener("grid-break-permanent", handleGridBreakPermanent);
      window.removeEventListener("grid-clear-group", handleGridClearGroup);
    };
  }, [isMobile, mouseRef, handleClick, handleGridBreak, handleGridBreakPermanent, handleGridClearGroup]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ width: "100%", height: "100%" }}
    />
  );
}
