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
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          position: "relative",
        }}
      >
        {/* Circle */}
        <div
          style={{
            width: 24,
            height: 24,
            borderRadius: "50%",
            border: "2px solid #fff",
            display: "flex",
          }}
        />
        {/* Horizontal line */}
        <div
          style={{
            position: "absolute",
            top: 15,
            left: 2,
            width: 28,
            height: 2,
            background: "#fff",
            display: "flex",
          }}
        />
        {/* Vertical line */}
        <div
          style={{
            position: "absolute",
            top: 2,
            left: 15,
            width: 2,
            height: 28,
            background: "#fff",
            display: "flex",
          }}
        />
      </div>
    ),
    { ...size }
  );
}
