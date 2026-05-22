import { cn } from "@/lib/utils"

export function ScoreBar({
  value,
  side = "a",
  className,
}: {
  value: number
  side?: "a" | "b"
  className?: string
}) {
  const pct = Math.max(0, Math.min(100, value))
  return (
    <div
      className={cn(
        "h-1 w-full overflow-hidden rounded-full bg-arena-neutral-soft",
        className,
      )}
    >
      <span
        className={cn(
          "block h-full rounded-full",
          side === "a"
            ? "bg-[linear-gradient(90deg,#FF5A1F_0%,#E84818_100%)]"
            : "bg-[linear-gradient(90deg,#00A86B_0%,#1ED48D_100%)]",
        )}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

export function ScoreRow({
  label,
  value,
  side = "a",
}: {
  label: string
  value: number
  side?: "a" | "b"
}) {
  return (
    <div className="grid grid-cols-[1fr_100px] items-center gap-2 py-1 text-xs">
      <div className="flex items-center justify-between gap-2">
        <span className="text-arena-text-secondary">{label}</span>
        <span className="font-mono text-arena-text-primary">{value.toFixed(1)}</span>
      </div>
      <ScoreBar value={value * 10} side={side} />
    </div>
  )
}
