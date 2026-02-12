import Link from "next/link";

export function Footer() {
  return (
    <footer className="relative z-10 mt-32 border-t border-white/5">
      <div className="mx-auto max-w-7xl px-6 sm:px-8 py-12">
        {/* Financial Disclaimer */}
        <div className="mb-10 rounded-2xl border border-white/5 bg-white/[0.02] p-6">
          <p className="text-[13px] leading-relaxed text-white/40">
            <span className="font-medium text-white/60">Disclaimer:</span>{" "}
            Cryptosensus provides data-driven onchain analysis for informational
            purposes only. This is not financial advice. Cryptocurrency trading
            involves substantial risk of loss. Past performance does not guarantee
            future results. Always do your own research (DYOR) and consult a
            licensed financial advisor before making investment decisions.
          </p>
        </div>

        {/* Links */}
        <div className="flex flex-wrap justify-center gap-x-12 gap-y-6 mb-10">
          <div className="text-center">
            <h3 className="text-[11px] font-medium uppercase tracking-wider text-white/20 mb-3">
              Legal
            </h3>
            <ul className="flex gap-6">
              <li>
                <Link
                  href="/disclaimer"
                  className="text-sm text-white/30 hover:text-white/60 transition-colors"
                >
                  Disclaimer
                </Link>
              </li>
              <li>
                <Link
                  href="/privacy"
                  className="text-sm text-white/30 hover:text-white/60 transition-colors"
                >
                  Privacy
                </Link>
              </li>
              <li>
                <Link
                  href="/terms"
                  className="text-sm text-white/30 hover:text-white/60 transition-colors"
                >
                  Terms
                </Link>
              </li>
            </ul>
          </div>
          <div className="text-center">
            <h3 className="text-[11px] font-medium uppercase tracking-wider text-white/20 mb-3">
              Connect
            </h3>
            <a
              href="https://x.com/Maximus0Primus"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-white/30 hover:text-white/60 transition-colors"
            >
              Twitter/X
            </a>
          </div>
        </div>

        {/* Copyright */}
        <div className="pt-6 border-t border-white/5 text-center">
          <p className="text-xs text-white/15">
            &copy; {new Date().getFullYear()} Cryptosensus. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}
