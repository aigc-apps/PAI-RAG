import { cn } from "@/lib/utils"

type StatusKind = "ok" | "warn" | "err" | "run" | "idle"

const KIND_STYLES: Record<StatusKind, string> = {
  ok: "bg-arena-success ring-arena-success-soft",
  warn: "bg-arena-warning ring-arena-warning-soft",
  err: "bg-arena-danger ring-arena-danger-soft",
  run: "bg-arena-info ring-arena-info-soft animate-arena-pulse",
  idle: "bg-arena-neutral ring-arena-neutral-soft",
}

export function StatusDot({
  kind,
  children,
  className,
}: {
  kind: StatusKind
  children?: React.ReactNode
  className?: string
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 text-xs text-arena-text-secondary",
        className,
      )}
    >
      <span
        className={cn(
          "inline-block size-1.5 rounded-full ring-[3px]",
          KIND_STYLES[kind],
        )}
      />
      {children}
    </span>
  )
}
