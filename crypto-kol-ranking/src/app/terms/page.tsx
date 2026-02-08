"use client";

import { Header } from "@/components/layout/Header";
import { FloatingNav } from "@/components/layout/FloatingNav";

export default function TermsPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="fixed inset-0 bg-grid-pattern pointer-events-none" />
      <Header />
      <FloatingNav />

      <main className="relative z-10 mx-auto max-w-3xl px-6 sm:px-8 pt-32 pb-40">
        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-white mb-12">
          Terms of Service
        </h1>

        <div className="space-y-10 text-[15px] leading-relaxed text-white/60">
          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Acceptance of Terms
            </h2>
            <p>
              By accessing or using Cryptosensus (&ldquo;the Platform&rdquo;),
              you agree to be bound by these Terms of Service. If you do not
              agree to these terms, do not use the Platform. We reserve the
              right to modify these terms at any time. Continued use after
              changes constitutes acceptance.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Service Description
            </h2>
            <p>
              Cryptosensus is a crypto sentiment analysis platform that
              aggregates and analyzes data from publicly available sources to
              generate token rankings and conviction scores. The Platform
              provides data-driven insights for informational purposes. It does
              not provide financial advice, portfolio management, or trading
              execution services.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              No Guarantees
            </h2>
            <p>
              The Platform is provided on an &ldquo;as is&rdquo; and &ldquo;as
              available&rdquo; basis. We make no guarantees regarding:
            </p>
            <ul className="list-disc list-inside mt-3 space-y-2">
              <li>
                The accuracy, completeness, or timeliness of any data or scores
              </li>
              <li>Uninterrupted or error-free service availability</li>
              <li>The financial outcome of any decisions based on our data</li>
              <li>The behavior or reliability of third-party data sources</li>
            </ul>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              User Conduct
            </h2>
            <p>You agree not to:</p>
            <ul className="list-disc list-inside mt-3 space-y-2">
              <li>
                Use the Platform for any unlawful purpose
              </li>
              <li>
                Attempt to access, tamper with, or exploit non-public areas of
                the Platform or its systems
              </li>
              <li>
                Scrape, crawl, or automate data extraction beyond normal usage
              </li>
              <li>
                Redistribute or resell Platform data without written permission
              </li>
              <li>
                Interfere with or disrupt the Platform or its infrastructure
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Intellectual Property
            </h2>
            <p>
              All content, design, algorithms, scoring methodologies, and code
              on the Platform are the intellectual property of Cryptosensus. You
              may not reproduce, distribute, or create derivative works from our
              content without prior written consent. Token names and symbols
              referenced on the Platform belong to their respective projects.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Limitation of Liability
            </h2>
            <p>
              To the maximum extent permitted by law, Cryptosensus and its
              creators shall not be liable for any indirect, incidental,
              special, consequential, or punitive damages, including but not
              limited to loss of profits, data, or other intangible losses
              resulting from your use of or inability to use the Platform.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Termination
            </h2>
            <p>
              We reserve the right to suspend or terminate access to the
              Platform at any time, for any reason, without prior notice. Upon
              termination, your right to use the Platform ceases immediately.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-4">
              Governing Law
            </h2>
            <p>
              These Terms shall be governed by and construed in accordance with
              applicable laws. Any disputes arising from these Terms or your
              use of the Platform shall be resolved through good-faith
              negotiation before pursuing formal legal proceedings.
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
