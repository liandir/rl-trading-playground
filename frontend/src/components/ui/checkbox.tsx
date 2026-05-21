"use client";

import * as React from "react";
import { Check } from "lucide-react";
import { cn } from "@/lib/cn";

export const Checkbox = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, checked, onChange, ...props }, ref) => (
  <label className="relative inline-flex items-center">
    <input
      ref={ref}
      type="checkbox"
      checked={checked}
      onChange={onChange}
      className="peer absolute h-4 w-4 cursor-pointer opacity-0"
      {...props}
    />
    <span
      className={cn(
        "grid h-4 w-4 place-items-center rounded border border-border bg-background transition-colors",
        "peer-checked:border-primary peer-checked:bg-primary peer-focus-visible:ring-2 peer-focus-visible:ring-primary/40",
        className
      )}
    >
      {checked && <Check className="h-3 w-3 text-primary-foreground" strokeWidth={3} />}
    </span>
  </label>
));
Checkbox.displayName = "Checkbox";
