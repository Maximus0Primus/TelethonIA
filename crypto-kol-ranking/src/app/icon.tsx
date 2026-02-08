import { ImageResponse } from "next/og";

export const runtime = "edge";
export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: 32,
          height: 32,
          background: "#000",
          borderRadius: 8,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Subtle green accent dot */}
        <div
          style={{
            position: "absolute",
            width: 12,
            height: 12,
            borderRadius: "50%",
            background: "radial-gradient(circle, rgba(0,255,65,0.4) 0%, transparent 70%)",
            top: 2,
            right: 2,
            display: "flex",
          }}
        />
        {/* C letter */}
        <div
          style={{
            fontSize: 22,
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
