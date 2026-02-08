import { ImageResponse } from "next/og";

export const runtime = "edge";
export const size = { width: 180, height: 180 };
export const contentType = "image/png";

export default function AppleIcon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: 180,
          height: 180,
          background: "#000",
          borderRadius: 40,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Grid pattern */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px)",
            backgroundSize: "20px 20px",
            display: "flex",
          }}
        />
        {/* Green accent glow */}
        <div
          style={{
            position: "absolute",
            width: 60,
            height: 60,
            borderRadius: "50%",
            background: "radial-gradient(circle, rgba(0,255,65,0.25) 0%, transparent 70%)",
            top: 10,
            right: 10,
            display: "flex",
          }}
        />
        {/* C letter */}
        <div
          style={{
            fontSize: 120,
            fontWeight: 800,
            color: "#fff",
            lineHeight: 1,
            display: "flex",
          }}
        >
          C
        </div>
      </div>
    ),
    { ...size }
  );
}
