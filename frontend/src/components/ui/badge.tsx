import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/cn";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium",
  {
    variants: {
      variant: {
        default: "bg-muted text-foreground border-border",
        primary: "bg-primary/10 text-primary border-primary/20",
        success: "bg-success/15 text-success border-success/30",
        warning: "bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30",
        destructive: "bg-destructive/15 text-destructive border-destructive/30",
        outline: "bg-transparent text-muted-foreground border-border",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}
