import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "Cryptosensus — Real-Time Lowcap Crypto Buy Indicator";
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
              "radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%)",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            display: "flex",
          }}
        />

        {/* Eye logo */}
        <svg
          width="120"
          height="85"
          viewBox="0 0 100 60"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          style={{ marginBottom: 30 }}
        >
          <path
            d="M50 2 L95 30 L50 58 L5 30 Z"
            stroke="white"
            strokeWidth="4"
            fill="none"
          />
          <path
            d="M50 10 L85 30 L50 50 L15 30 Z"
            stroke="white"
            strokeWidth="3"
            fill="none"
          />
          <circle cx="50" cy="30" r="10" stroke="white" strokeWidth="3.5" fill="none" />
        </svg>

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
              background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)",
              display: "flex",
            }}
          />

          <div
            style={{
              fontSize: 28,
              color: "rgba(255,255,255,0.5)",
              letterSpacing: "4px",
              textTransform: "uppercase",
              display: "flex",
            }}
          >
            Real-Time Lowcap Buy Indicator
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
              fontSize: 18,
              color: "rgba(255,255,255,0.3)",
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
