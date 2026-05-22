import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-arena-sm border px-1.5 py-0.5 font-mono text-[10.5px] font-semibold uppercase tracking-wider transition-colors focus:outline-none focus:ring-2 focus:ring-arena-accent",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-arena-accent-soft text-arena-accent-press",
        accent:
          "border-transparent bg-arena-accent text-white",
        secondary:
          "border-transparent bg-arena-neutral-soft text-arena-text-secondary",
        neutral:
          "border-transparent bg-arena-neutral-soft text-arena-text-secondary",
        outline:
          "border-arena-border-strong bg-transparent text-arena-text-secondary",
        success:
          "border-transparent bg-arena-success-soft text-arena-success",
        warning:
          "border-transparent bg-arena-warning-soft text-arena-warning",
        destructive:
          "border-transparent bg-arena-danger-soft text-arena-danger",
        info:
          "border-transparent bg-arena-info-soft text-arena-info",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
