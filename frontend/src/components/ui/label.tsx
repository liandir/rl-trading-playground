import * as React from "react";
import { cn } from "@/lib/cn";

export const Label = React.forwardRef<HTMLLabelElement, React.LabelHTMLAttributes<HTMLLabelElement>>(
  ({ className, ...props }, ref) => (
    <label
      ref={ref}
      className={cn(
        "text-xs font-medium uppercase tracking-wide text-muted-foreground",
        className
      )}
      {...props}
    />
  )
);
Label.displayName = "Label";
