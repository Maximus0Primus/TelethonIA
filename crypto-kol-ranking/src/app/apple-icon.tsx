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
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          position: "relative",
        }}
      >
        {/* Circle */}
        <div
          style={{
            width: 130,
            height: 130,
            borderRadius: "50%",
            border: "8px solid #fff",
            display: "flex",
          }}
        />
        {/* Horizontal line */}
        <div
          style={{
            position: "absolute",
            top: 86,
            left: 10,
            width: 160,
            height: 8,
            background: "#fff",
            display: "flex",
          }}
        />
        {/* Vertical line */}
        <div
          style={{
            position: "absolute",
            top: 10,
            left: 86,
            width: 8,
            height: 160,
            background: "#fff",
            display: "flex",
          }}
        />
      </div>
    ),
    { ...size }
  );
}
