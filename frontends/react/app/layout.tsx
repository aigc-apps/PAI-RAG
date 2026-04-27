import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PAI Assistant",
  description: "Web client for the PAI-RAG agent backend.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
