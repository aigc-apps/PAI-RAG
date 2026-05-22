import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"

export function CodeBlock({ value, compact = false }: { value: string; compact?: boolean }) {
  return (
    <ScrollArea className={cn(compact ? "h-[180px]" : "h-[240px]", "rounded-md border bg-slate-950 p-4")}>
      <pre className="whitespace-pre-wrap break-words text-xs leading-5 text-slate-100">{value}</pre>
    </ScrollArea>
  )
}
