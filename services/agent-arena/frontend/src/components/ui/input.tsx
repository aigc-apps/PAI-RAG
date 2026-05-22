import * as React from "react"

import { cn } from "@/lib/utils"

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-9 w-full rounded-arena border border-arena-border-strong bg-white px-3 py-2 text-[13px] text-arena-text-primary",
          "placeholder:text-arena-text-tertiary",
          "focus:outline-none focus:border-arena-accent focus:ring-2 focus:ring-arena-accent/25",
          "disabled:cursor-not-allowed disabled:bg-arena-bg-subtle disabled:opacity-60",
          "file:border-0 file:bg-transparent file:text-sm file:font-medium",
          className,
        )}
        ref={ref}
        {...props}
      />
    )
  },
)
Input.displayName = "Input"

export { Input }
