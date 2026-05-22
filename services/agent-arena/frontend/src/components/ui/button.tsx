import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-arena text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-arena-accent focus-visible:ring-offset-1 focus-visible:ring-offset-arena-bg-page disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default:
          "bg-arena-accent text-white shadow-[0_1px_0_rgba(0,0,0,0.04),inset_0_1px_0_rgba(255,255,255,0.18)] hover:bg-arena-accent-hover active:bg-arena-accent-press",
        destructive:
          "bg-arena-danger text-white hover:bg-arena-danger/90 active:bg-arena-danger/80",
        outline:
          "border border-arena-border-strong bg-white text-arena-text-primary hover:border-arena-text-tertiary hover:bg-arena-bg-hover",
        secondary:
          "border border-arena-border bg-arena-bg-subtle text-arena-text-primary hover:bg-arena-bg-hover",
        ghost:
          "text-arena-text-secondary hover:bg-arena-bg-hover hover:text-arena-text-primary",
      },
      size: {
        default: "h-9 px-3.5 py-2 text-[13px]",
        sm: "h-8 rounded-arena px-3 text-xs",
        lg: "h-10 rounded-arena px-6 text-sm",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  },
)
Button.displayName = "Button"

export { Button, buttonVariants }
