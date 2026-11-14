import type { Metadata } from "next";
import "./globals.css";
import "./benchmark.css";

export const metadata: Metadata = {
  title: "GroundTruth AI | Retrieval Workbench",
  description: "Explore permission-aware retrieval, grounded answers, and measurable RAG evaluations.",
  applicationName: "GroundTruth AI",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
