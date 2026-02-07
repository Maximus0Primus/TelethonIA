"use client";

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

interface ContactModalProps {
  trigger?: React.ReactNode;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export function ContactModal({ trigger, open: controlledOpen, onOpenChange }: ContactModalProps = {}) {
  const [internalOpen, setInternalOpen] = useState(false);
  const open = controlledOpen ?? internalOpen;
  const setOpen = onOpenChange ?? setInternalOpen;
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [status, setStatus] = useState<"idle" | "sending" | "sent" | "error">(
    "idle"
  );

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("sending");

    try {
      const res = await fetch("/api/contact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, message }),
      });

      if (!res.ok) throw new Error();

      setStatus("sent");
      setEmail("");
      setMessage("");
    } catch {
      setStatus("error");
    }
  }

  function handleOpenChange(next: boolean) {
    setOpen(next);
    if (!next) setStatus("idle");
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        {trigger ?? (
          <button className="pointer-events-auto rounded-full border border-white bg-white px-4 py-1.5 font-mono text-xs tracking-wider text-black hover:bg-white/85 transition-colors uppercase">
            Let&apos;s Talk
          </button>
        )}
      </DialogTrigger>

      <DialogContent className="glass-modal border-white/10 sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="font-mono text-sm tracking-wider text-white/90 uppercase">
            Get in touch
          </DialogTitle>
        </DialogHeader>

        {status === "sent" ? (
          <div className="py-8 text-center">
            <p className="font-mono text-sm text-white/70">
              Message sent. I&apos;ll get back to you.
            </p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <input
              type="email"
              placeholder="your@email.com (optional)"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-md border border-white/10 bg-white/5 px-3 py-2 font-mono text-sm text-white/90 placeholder:text-white/30 outline-none focus:border-white/25 transition-colors"
            />
            <textarea
              required
              placeholder="Your message..."
              rows={4}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              className="w-full resize-none rounded-md border border-white/10 bg-white/5 px-3 py-2 font-mono text-sm text-white/90 placeholder:text-white/30 outline-none focus:border-white/25 transition-colors"
            />

            {status === "error" && (
              <p className="font-mono text-xs text-red-400">
                Failed to send. Try again.
              </p>
            )}

            <button
              type="submit"
              disabled={status === "sending"}
              className="w-full rounded-md border border-white/10 bg-white/10 py-2 font-mono text-xs tracking-wider text-white/80 uppercase hover:bg-white/15 transition-colors disabled:opacity-40"
            >
              {status === "sending" ? "Sending..." : "Send"}
            </button>
          </form>
        )}

        <div className="flex flex-col gap-2 border-t border-white/10 pt-4">
          <span className="font-mono text-[11px] text-white/40">
            Contact :
          </span>
          <div className="flex items-center gap-3">
            <a
              href="https://x.com/Maximus0Primus"
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-[11px] text-white/50 hover:text-white/80 transition-colors"
            >
              𝕏 @S£igneur
            </a>
            <span className="text-white/20">|</span>
            <a
              href="https://t.me/Maximus0Primus"
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-[11px] text-white/50 hover:text-white/80 transition-colors"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" className="inline size-3"><path d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.479.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z"/></svg> @S£igneur
            </a>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
