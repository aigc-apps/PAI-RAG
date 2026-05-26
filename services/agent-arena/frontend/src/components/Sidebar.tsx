import {
  Activity,
  FileText,
  FlaskConical,
  Gauge,
  History,
  Server,
  Settings2,
  Swords,
} from "lucide-react"

import { cn } from "@/lib/utils"
import type { ViewMode } from "@/lib/types"

type NavItem = {
  key: ViewMode | string
  label: string
  icon: React.ReactNode
  tag?: string
  disabled?: boolean
}

const NAV_PRIMARY: NavItem[] = [
  { key: "arena", label: "竞技场对比", icon: <Swords className="size-4" />, tag: "A/B" },
  { key: "batch", label: "稳定性测试", icon: <FlaskConical className="size-4" />, tag: "N×" },
  { key: "history", label: "历史记录", icon: <History className="size-4" /> },
]

const NAV_RESOURCES: NavItem[] = [
  { key: "endpoints", label: "Agent 端点", icon: <Server className="size-4" /> },
  { key: "datasets", label: "评测集", icon: <FileText className="size-4" /> },
  { key: "judge-model", label: "Judge 模型", icon: <Settings2 className="size-4" /> },
]

const NAV_OPS: NavItem[] = [
  { key: "logs", label: "调用日志", icon: <Activity className="size-4" />, disabled: true },
  { key: "health", label: "健康监控", icon: <Gauge className="size-4" />, disabled: true },
]

export function Sidebar({
  view,
  onViewChange,
  historyCount,
}: {
  view: ViewMode
  onViewChange: (view: ViewMode) => void
  historyCount?: number
}) {
  return (
    <aside className="fixed bottom-0 left-0 top-topbar flex w-sidebar flex-col overflow-y-auto border-r border-arena-border bg-arena-bg-sidebar py-4">
      <NavSection label="评测">
        {NAV_PRIMARY.map((item) => {
          const isHistory = item.key === "history"
          const tag = isHistory && historyCount !== undefined ? String(historyCount) : item.tag
          return (
            <NavRow
              key={item.key}
              item={{ ...item, tag }}
              active={item.key === view}
              onClick={() => onViewChange(item.key as ViewMode)}
            />
          )
        })}
      </NavSection>
      <NavSection label="资源">
        {NAV_RESOURCES.map((item) => (
          <NavRow
            key={item.key}
            item={item}
            active={item.key === view}
            onClick={() => onViewChange(item.key as ViewMode)}
          />
        ))}
      </NavSection>
      <NavSection label="运维">
        {NAV_OPS.map((item) => (
          <NavRow key={item.key} item={item} />
        ))}
      </NavSection>
      <div className="mt-auto border-t border-arena-border px-6 py-3.5 font-mono text-[11px] text-arena-text-tertiary">
        v0.5.2 · build a1c4f9d
      </div>
    </aside>
  )
}

function NavSection({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-4 px-3">
      <div className="px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.06em] text-arena-text-tertiary">
        {label}
      </div>
      {children}
    </div>
  )
}

function NavRow({
  item,
  active,
  onClick,
}: {
  item: NavItem
  active?: boolean
  onClick?: () => void
}) {
  return (
    <button
      type="button"
      onClick={item.disabled ? undefined : onClick}
      disabled={item.disabled}
      className={cn(
        "relative mb-0.5 flex w-full items-center gap-2.5 rounded-arena px-3 py-2 text-left text-[13px] font-medium transition",
        active
          ? "bg-arena-accent-soft font-semibold text-arena-accent-press"
          : "text-arena-text-secondary hover:bg-arena-bg-hover hover:text-arena-text-primary",
        item.disabled && "cursor-not-allowed opacity-50 hover:bg-transparent hover:text-arena-text-secondary",
      )}
    >
      {active ? (
        <span className="absolute inset-y-2 -left-3 w-[3px] rounded-r-sm bg-arena-accent" />
      ) : null}
      <span className="shrink-0">{item.icon}</span>
      <span className="flex-1 truncate">{item.label}</span>
      {item.tag ? (
        <span
          className={cn(
            "ml-auto rounded-full px-1.5 py-px font-mono text-[10px] font-semibold",
            active
              ? "bg-arena-accent-tint text-arena-accent-press"
              : "bg-arena-border text-arena-text-tertiary",
          )}
        >
          {item.tag}
        </span>
      ) : null}
    </button>
  )
}
