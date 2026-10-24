import type { SVGProps } from "react";

const PATHS = {
  search: "M11 4a7 7 0 1 0 4.2 12.6l4.1 4.1 1.4-1.4-4.1-4.1A7 7 0 0 0 11 4Zm0 2a5 5 0 1 1 0 10 5 5 0 0 1 0-10Z",
  plus: "M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6z",
  minus: "M5 11h14v2H5z",
  rotateLeft: "M12 5a7 7 0 1 1-6.6 9.3l1.9-.6A5 5 0 1 0 8.5 8.5L11 11H4V4l3.1 3.1A7 7 0 0 1 12 5Z",
  rotateRight: "M12 5a7 7 0 1 0 6.6 9.3l-1.9-.6a5 5 0 1 1-1.2-5.2L13 11h7V4l-3.1 3.1A7 7 0 0 0 12 5Z",
  home: "M12 3 3 10.5V21h7v-6h4v6h7V10.5L12 3Zm0 2.6 7 5.8V19h-3v-6H8v6H5v-7.6l7-5.8Z",
  bell: "M12 3a6 6 0 0 0-6 6v4l-2 3v1h16v-1l-2-3V9a6 6 0 0 0-6-6Zm-2 16a2 2 0 0 0 4 0h-4Z",
  branch: "M7 3a3 3 0 0 0-1 5.8v6.4A3 3 0 1 0 8 15.2V13a2 2 0 0 1 2-2h4a4 4 0 0 0 4-4v-.2a3 3 0 1 0-2 0V7a2 2 0 0 1-2 2h-4a4 4 0 0 0-2 .5V8.8A3 3 0 0 0 7 3Z",
  box: "M12 2 3 7v10l9 5 9-5V7l-9-5Zm0 2.3L18.6 8 12 11.6 5.4 8 12 4.3ZM5 9.7l6 3.3v6.6l-6-3.3V9.7Zm8 9.9V13l6-3.3v6.6l-6 3.3Z",
  check: "M9.5 16.6 5.4 12.5 4 13.9l5.5 5.5L20 8.9l-1.4-1.4z",
  truck: "M2 6h12v9h1.2a3 3 0 0 1 5.6 0H22v-4l-3-4h-4V6H2v9h.2a3 3 0 0 1 5.6 0H14V6Zm16 9a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3ZM5 15a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm10-6h3l2 2.7V13h-5V9Z",
  crane: "M4 21V5h2v2h14v2H11l-3 3v9H4Zm2-12v1.2L7.2 9H6Zm11 1h2v6h-2v-6Z",
  flag: "M5 3h2v1h11l-2 4 2 4H7v9H5V3Z",
  play: "M8 5v14l11-7z",
  chevronLeft: "M15.4 6.6 14 5.2 7.2 12l6.8 6.8 1.4-1.4L10 12z",
  chevronRight: "M8.6 6.6 10 5.2l6.8 6.8-6.8 6.8-1.4-1.4L14 12z",
  close: "M6.4 5 5 6.4 10.6 12 5 17.6 6.4 19l5.6-5.6 5.6 5.6 1.4-1.4-5.6-5.6L19 6.4 17.6 5 12 10.6z",
  pin: "M12 2a7 7 0 0 0-7 7c0 5 7 13 7 13s7-8 7-13a7 7 0 0 0-7-7Zm0 4a3 3 0 1 1 0 6 3 3 0 0 1 0-6Z",
};

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 18, ...props }: { name: IconName; size?: number } & SVGProps<SVGSVGElement>) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" {...props}>
      <path d={PATHS[name]} />
    </svg>
  );
}
