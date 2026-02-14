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
        }}
      >
        <svg
          width="28"
          height="20"
          viewBox="0 0 100 60"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Outer eye shape - diamond/angular */}
          <path
            d="M50 2 L95 30 L50 58 L5 30 Z"
            stroke="white"
            strokeWidth="5"
            fill="none"
          />
          {/* Inner eye shape - slightly smaller */}
          <path
            d="M50 10 L85 30 L50 50 L15 30 Z"
            stroke="white"
            strokeWidth="4"
            fill="none"
          />
          {/* Pupil circle */}
          <circle cx="50" cy="30" r="10" stroke="white" strokeWidth="4" fill="none" />
        </svg>
      </div>
    ),
    { ...size }
  );
}
