import type { Metadata } from "next";
import { Inter } from "next/font/google";
import type { ReactNode } from "react";

import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "DevSite 開發進度工地",
  description: "以 3D 工地地圖呈現改善計畫、提交紀錄、測試健康度與交接狀態",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="zh-Hant" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
