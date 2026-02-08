"use client";

import { Header } from "@/components/layout/Header";
import { FloatingNav } from "@/components/layout/FloatingNav";

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="fixed inset-0 bg-grid-pattern pointer-events-none" />
      <Header />
      <FloatingNav />

      <main className="relative z-10 mx-auto max-w-3xl px-6 sm:px-8 pt-32 pb-40">
        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-white mb-12">
          Privacy Policy
        </h1>

        <div className="space-y-10 text-[15px] leading-relaxed text-white/60">
          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Information We Collect
            </h2>
            <p>
              Cryptosensus collects minimal personal data. We may collect:
            </p>
            <ul className="list-disc list-inside mt-3 space-y-2">
              <li>
                Email address if you contact us or subscribe to updates
              </li>
              <li>
                Usage data such as pages visited, time spent, and interactions
                (collected via analytics)
              </li>
              <li>
                Device and browser information for performance optimization
              </li>
            </ul>
            <p className="mt-3">
              We do not require account creation to use the platform. We do not
              collect wallet addresses, trading history, or financial data from
              users.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              How We Use Your Information
            </h2>
            <p>Any information collected is used to:</p>
            <ul className="list-disc list-inside mt-3 space-y-2">
              <li>Improve the platform experience and performance</li>
              <li>Respond to support inquiries</li>
              <li>Analyze aggregate usage patterns</li>
              <li>Send service-related communications (if subscribed)</li>
            </ul>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Third-Party Services
            </h2>
            <p>
              Cryptosensus uses the following third-party services that may
              process data on our behalf:
            </p>
            <ul className="list-disc list-inside mt-3 space-y-2">
              <li>
                <span className="text-white/80">Vercel</span> — Hosting and
                deployment
              </li>
              <li>
                <span className="text-white/80">Supabase</span> — Database and
                authentication infrastructure
              </li>
            </ul>
            <p className="mt-3">
              Each of these services has its own privacy policy governing data
              processing. We encourage you to review their policies.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Cookies
            </h2>
            <p>
              We may use essential cookies to maintain session state and
              preferences. We do not use third-party tracking cookies for
              advertising purposes. You can configure your browser to reject
              cookies, though this may affect platform functionality.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Data Sharing
            </h2>
            <p>
              We do not sell, trade, or rent your personal information to third
              parties. We may share data only in the following circumstances:
            </p>
            <ul className="list-disc list-inside mt-3 space-y-2">
              <li>When required by law or legal process</li>
              <li>To protect our rights or property</li>
              <li>
                With service providers who assist in operating the platform
                (under strict data processing agreements)
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Your Rights
            </h2>
            <p>
              Depending on your jurisdiction, you may have the right to access,
              correct, delete, or port your personal data. To exercise these
              rights, contact us at the email below. We will respond to requests
              within 30 days.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Contact
            </h2>
            <p>
              For privacy-related inquiries, reach us via the contact form on
              the platform or through our{" "}
              <a
                href="https://x.com/Maximus0Primus"
                target="_blank"
                rel="noopener noreferrer"
                className="text-white/80 underline hover:text-white transition-colors"
              >
                Twitter/X
              </a>{" "}
              account.
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
