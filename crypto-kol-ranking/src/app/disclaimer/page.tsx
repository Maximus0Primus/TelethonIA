"use client";

import { Header } from "@/components/layout/Header";
import { FloatingNav } from "@/components/layout/FloatingNav";

export default function DisclaimerPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="fixed inset-0 bg-grid-pattern pointer-events-none" />
      <Header />
      <FloatingNav />

      <main className="relative z-10 mx-auto max-w-3xl px-6 sm:px-8 pt-32 pb-40">
        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-white mb-12">
          Disclaimer
        </h1>

        <div className="space-y-10 text-[15px] leading-relaxed text-white/60">
          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              No Financial Advice
            </h2>
            <p>
              Cryptosensus is a data analysis platform that provides sentiment
              scoring based on publicly available information. Nothing on this
              website constitutes financial, investment, legal, or tax advice.
              The content provided is for informational purposes only and should
              not be construed as a recommendation to buy, sell, or hold any
              cryptocurrency or digital asset.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Risk Warning
            </h2>
            <p>
              Cryptocurrency trading involves substantial risk of loss and is not
              suitable for every investor. The value of digital assets can
              fluctuate dramatically in short periods. You should only invest
              money you can afford to lose entirely. Before making any investment
              decisions, consult with a qualified financial advisor who
              understands your individual circumstances.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Data Accuracy
            </h2>
            <p>
              While we strive to provide accurate and timely data, Cryptosensus
              makes no warranties or representations regarding the completeness,
              accuracy, or reliability of any information displayed on this
              platform. Sentiment scores, rankings, and analytics are generated
              algorithmically and may contain errors or inaccuracies. Data is
              provided &ldquo;as is&rdquo; without any guarantee of correctness.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Do Your Own Research (DYOR)
            </h2>
            <p>
              Always conduct your own independent research before making any
              financial decisions. The scores and rankings on Cryptosensus
              reflect algorithmic analysis of social sentiment and should be
              considered one data point among many in your research process. Past
              sentiment patterns do not guarantee future token performance.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Limitation of Liability
            </h2>
            <p>
              Cryptosensus, its creators, contributors, and affiliates shall not
              be held liable for any losses, damages, or claims arising from the
              use of this platform or reliance on the information provided.
              Users assume full responsibility for their trading and investment
              decisions.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Third-Party Content
            </h2>
            <p>
              Cryptosensus aggregates data from third-party sources including
              public Telegram groups. We do not endorse, verify, or take
              responsibility for the accuracy of statements made by third-party
              Key Opinion Leaders (KOLs) or any other external sources. The
              inclusion of any token in our rankings does not constitute an
              endorsement.
            </p>
          </section>

          <p className="text-white/30 text-sm pt-8 border-t border-white/5">
            Last updated: February 2026
          </p>
        </div>
      </main>
    </div>
  );
}
