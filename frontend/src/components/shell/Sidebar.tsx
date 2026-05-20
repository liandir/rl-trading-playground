"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  Bot,
  CandlestickChart,
  Database,
  GaugeCircle,
  LineChart,
  Server,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/cn";

type NavItem = {
  href: string;
  label: string;
  icon: LucideIcon;
  description?: string;
};

const NAV: NavItem[] = [
  { href: "/", label: "Dashboard", icon: GaugeCircle },
  { href: "/runs", label: "Runs", icon: Activity },
  { href: "/agents", label: "Agents", icon: Bot },
  { href: "/envs", label: "Environments", icon: Server },
  { href: "/data", label: "Data", icon: Database },
  { href: "/validation", label: "Validation", icon: LineChart },
  { href: "/live", label: "Live", icon: CandlestickChart },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="hidden h-screen w-60 shrink-0 border-r border-border bg-card/40 px-3 py-5 md:flex md:flex-col">
      <Link href="/" className="px-3 py-2 mb-4 group">
        <div className="flex items-center gap-2">
          <div className="grid h-8 w-8 place-items-center rounded-md bg-primary text-primary-foreground">
            <CandlestickChart className="h-4 w-4" />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-semibold leading-none">Trading Studio</span>
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground">RL ops</span>
          </div>
        </div>
      </Link>
      <nav className="flex flex-1 flex-col gap-1">
        {NAV.map((item) => {
          const active = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "group flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
                active
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
            >
              <Icon className="h-4 w-4" />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
      <div className="px-3 py-2 text-[10px] uppercase tracking-wide text-muted-foreground">
        v0.1.0
      </div>
    </aside>
  );
}
