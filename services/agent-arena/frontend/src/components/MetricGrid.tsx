import { cn } from "@/lib/utils"

export type MetricCell = {
  label: string
  value: React.ReactNode
  sub?: React.ReactNode
  tone?: "default" | "muted" | "danger"
}

const TONE: Record<NonNullable<MetricCell["tone"]>, string> = {
  default: "text-arena-text-primary",
  muted: "text-arena-text-tertiary",
  danger: "text-arena-danger",
}

export function MetricGrid({
  items,
  columns = 4,
  className,
}: {
  items: MetricCell[]
  columns?: 2 | 3 | 4
  className?: string
}) {
  const cols =
    columns === 2 ? "grid-cols-2" : columns === 3 ? "grid-cols-3" : "grid-cols-4"
  return (
    <div
      className={cn(
        "grid divide-x divide-arena-border border-y border-arena-border",
        cols,
        className,
      )}
    >
      {items.map((cell, index) => (
        <div key={`${cell.label}-${index}`} className="px-3.5 py-2.5">
          <div className="text-[10.5px] font-semibold uppercase tracking-[0.04em] text-arena-text-tertiary">
            {cell.label}
          </div>
          <div
            className={cn(
              "mt-0.5 font-mono text-[16px] font-semibold leading-tight tracking-tight",
              TONE[cell.tone ?? "default"],
            )}
          >
            {cell.value}
            {cell.sub ? (
              <span className="ml-0.5 font-mono text-[11px] text-arena-text-tertiary">
                {cell.sub}
              </span>
            ) : null}
          </div>
        </div>
      ))}
    </div>
  )
}
