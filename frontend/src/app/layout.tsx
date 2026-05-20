import type { Metadata } from "next";
import "./globals.css";
import "uplot/dist/uPlot.min.css";
import { Sidebar } from "@/components/shell/Sidebar";
import { TopBar } from "@/components/shell/TopBar";
import { QueryProvider } from "@/components/shell/QueryProvider";

export const metadata: Metadata = {
  title: "Trading Studio",
  description: "RL trading experiment management and live monitoring.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased">
        <QueryProvider>
          <div className="flex min-h-screen">
            <Sidebar />
            <div className="flex min-w-0 flex-1 flex-col">
              <TopBar />
              <main className="flex-1 overflow-x-hidden px-6 py-6 md:px-10">
                <div className="mx-auto max-w-7xl">{children}</div>
              </main>
            </div>
          </div>
        </QueryProvider>
      </body>
    </html>
  );
}
