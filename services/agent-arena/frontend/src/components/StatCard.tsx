import { cn } from "@/lib/utils"

export type StatItem = {
  label: React.ReactNode
  value: React.ReactNode
  delta?: { text: React.ReactNode; tone?: "up" | "down" | "muted" }
  icon?: React.ReactNode
}

const DELTA_TONE = {
  up: "text-arena-success",
  down: "text-arena-danger",
  muted: "text-arena-text-tertiary",
} as const

export function StatCardRow({
  items,
  className,
}: {
  items: StatItem[]
  className?: string
}) {
  return (
    <div
      className={cn(
        "grid divide-x divide-arena-border overflow-hidden rounded-arena-lg border border-arena-border bg-arena-bg-card",
        items.length === 3
          ? "grid-cols-3"
          : items.length === 2
            ? "grid-cols-2"
            : "grid-cols-4",
        className,
      )}
    >
      {items.map((item, index) => (
        <div key={index} className="px-4 py-3.5">
          <div className="flex items-center gap-1.5 text-[11.5px] font-medium text-arena-text-tertiary">
            {item.icon}
            {item.label}
          </div>
          <div className="mt-1 font-mono text-[22px] font-bold leading-tight tracking-tight text-arena-text-primary">
            {item.value}
          </div>
          {item.delta ? (
            <div
              className={cn(
                "mt-0.5 font-mono text-[11px]",
                DELTA_TONE[item.delta.tone ?? "muted"],
              )}
            >
              {item.delta.text}
            </div>
          ) : null}
        </div>
      ))}
    </div>
  )
}
