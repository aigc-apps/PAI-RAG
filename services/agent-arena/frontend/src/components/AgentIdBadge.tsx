import { cn } from "@/lib/utils"

const GRADIENTS: Record<"a" | "b", string> = {
  a: "bg-[linear-gradient(135deg,#2A60E0_0%,#4F7DEA_100%)]",
  b: "bg-[linear-gradient(135deg,#00A86B_0%,#1ED48D_100%)]",
}

export function AgentIdBadge({
  id,
  size = "md",
  className,
}: {
  id: "a" | "b"
  size?: "sm" | "md"
  className?: string
}) {
  return (
    <span
      className={cn(
        "grid place-items-center rounded-md font-mono font-bold text-white shadow-sm",
        size === "sm" ? "size-5 text-[10px]" : "size-6 text-[13px]",
        GRADIENTS[id],
        className,
      )}
    >
      {id.toUpperCase()}
    </span>
  )
}
