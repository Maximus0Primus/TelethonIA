import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "Cryptosensus — Real-Time Crypto Lowcap Buy Score";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OGImage() {
  return new ImageResponse(
    (
      <div
        style={{
          background: "#000",
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Grid background */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)",
            backgroundSize: "60px 60px",
            display: "flex",
          }}
        />

        {/* Glow effect */}
        <div
          style={{
            position: "absolute",
            width: 500,
            height: 500,
            borderRadius: "50%",
            background:
              "radial-gradient(circle, rgba(138,43,226,0.15) 0%, transparent 70%)",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            display: "flex",
          }}
        />

        {/* Title */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 20,
            zIndex: 1,
          }}
        >
          <div
            style={{
              fontSize: 72,
              fontWeight: 800,
              color: "#fff",
              letterSpacing: "-2px",
              display: "flex",
            }}
          >
            CRYPTOSENSUS
          </div>

          {/* Divider line */}
          <div
            style={{
              width: 120,
              height: 2,
              background: "linear-gradient(90deg, transparent, #8a2be2, transparent)",
              display: "flex",
            }}
          />

          <div
            style={{
              fontSize: 28,
              color: "rgba(255,255,255,0.6)",
              letterSpacing: "4px",
              textTransform: "uppercase",
              display: "flex",
            }}
          >
            Real-Time Lowcap Buy Score
          </div>
        </div>

        {/* Bottom tag */}
        <div
          style={{
            position: "absolute",
            bottom: 40,
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: "#00ff88",
              display: "flex",
            }}
          />
          <div
            style={{
              fontSize: 18,
              color: "rgba(255,255,255,0.4)",
              letterSpacing: "2px",
              display: "flex",
            }}
          >
            cryptosensus.org
          </div>
        </div>
      </div>
    ),
    { ...size }
  );
}
