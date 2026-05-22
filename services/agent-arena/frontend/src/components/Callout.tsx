import { cn } from "@/lib/utils"

type CalloutVariant = "warning" | "info" | "danger" | "success"

const VARIANTS: Record<
  CalloutVariant,
  { container: string; title: string }
> = {
  warning: {
    container: "border-arena-warning bg-arena-warning-soft text-[#6B4408]",
    title: "text-[#6B4408]",
  },
  info: {
    container: "border-arena-info bg-arena-info-soft text-[#1B3A8F]",
    title: "text-[#1B3A8F]",
  },
  danger: {
    container: "border-arena-danger bg-arena-danger-soft text-[#7A1027]",
    title: "text-[#7A1027]",
  },
  success: {
    container: "border-arena-success bg-arena-success-soft text-[#0A5D40]",
    title: "text-[#0A5D40]",
  },
}

export function Callout({
  variant = "warning",
  title,
  icon,
  children,
  className,
}: {
  variant?: CalloutVariant
  title?: React.ReactNode
  icon?: React.ReactNode
  children?: React.ReactNode
  className?: string
}) {
  const styles = VARIANTS[variant]
  return (
    <div
      className={cn(
        "rounded-arena border px-3.5 py-3 text-[12.5px]",
        styles.container,
        className,
      )}
    >
      {title ? (
        <div
          className={cn(
            "mb-1 flex items-center gap-1.5 font-bold",
            styles.title,
          )}
        >
          {icon}
          {title}
        </div>
      ) : null}
      {children}
    </div>
  )
}
