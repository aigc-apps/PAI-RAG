import * as React from "react"

import { cn } from "@/lib/utils"

export interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className={cn(
          "flex min-h-[96px] w-full rounded-arena border border-arena-border-strong bg-white px-3 py-2 text-[13px] leading-relaxed text-arena-text-primary",
          "placeholder:text-arena-text-tertiary",
          "focus:outline-none focus:border-arena-accent focus:ring-2 focus:ring-arena-accent/25",
          "disabled:cursor-not-allowed disabled:bg-arena-bg-subtle disabled:opacity-60",
          className,
        )}
        ref={ref}
        {...props}
      />
    )
  },
)
Textarea.displayName = "Textarea"

export { Textarea }
