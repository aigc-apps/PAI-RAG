import { Sidebar } from "@/components/Sidebar"
import { TopBar } from "@/components/TopBar"
import { cn } from "@/lib/utils"
import type { ViewMode } from "@/lib/types"

export function AppShell({
  view,
  onViewChange,
  configured,
  agentCount,
  historyCount,
  onAccessKey,
  children,
  className,
}: {
  view: ViewMode
  onViewChange: (view: ViewMode) => void
  configured: boolean
  agentCount?: number
  historyCount?: number
  onAccessKey?: () => void
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className="min-h-screen bg-arena-bg-page font-sans text-arena-text-primary">
      <TopBar
        view={view}
        configured={configured}
        agentCount={agentCount}
        onAccessKey={onAccessKey}
      />
      <Sidebar view={view} onViewChange={onViewChange} historyCount={historyCount} />
      <main
        className={cn(
          "ml-sidebar mt-topbar min-h-[calc(100vh-theme(spacing.topbar))]",
          className,
        )}
      >
        {children}
      </main>
    </div>
  )
}

export function PageHeader({
  title,
  badge,
  subtitle,
  actions,
  className,
}: {
  title: React.ReactNode
  badge?: React.ReactNode
  subtitle?: React.ReactNode
  actions?: React.ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        "flex items-start gap-4 border-b border-arena-border bg-white px-8 pb-4 pt-5",
        className,
      )}
    >
      <div className="min-w-0 flex-1">
        <div className="mb-1.5 flex items-center gap-3.5">
          <h1 className="text-xl font-bold tracking-tight text-arena-text-primary">
            {title}
          </h1>
          {badge ? (
            <span className="rounded-arena-sm bg-arena-accent-soft px-2 py-0.5 font-mono text-[11px] font-semibold uppercase tracking-wider text-arena-accent-press">
              {badge}
            </span>
          ) : null}
        </div>
        {subtitle ? (
          <p className="text-[13px] text-arena-text-tertiary">{subtitle}</p>
        ) : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </div>
  )
}

export function PageBody({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  return <div className={cn("px-8 pb-16 pt-6", className)}>{children}</div>
}
