import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Specify the root directory for Turbopack to avoid lockfile warnings
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
