import { Bell, BookOpen } from "lucide-react"

import { cn } from "@/lib/utils"
import type { ViewMode } from "@/lib/types"

const PAGE_LABEL: Record<ViewMode, string> = {
  arena: "竞技场对比",
  batch: "稳定性测试",
  history: "历史记录",
  endpoints: "Agent 端点",
  "judge-model": "Judge 模型",
  datasets: "评测集",
}

export function TopBar({
  view,
  configured,
  agentCount,
  onAccessKey,
}: {
  view: ViewMode
  configured: boolean
  agentCount?: number
  onAccessKey?: () => void
}) {
  return (
    <header className="fixed inset-x-0 top-0 z-50 flex h-topbar items-center border-b border-[#1A1F27] bg-arena-bg-topbar px-5 text-arena-text-inverse">
      <div className="flex items-center gap-2.5 text-sm font-bold tracking-tight">
        <div
          className={cn(
            "grid size-[26px] place-items-center rounded-md font-mono text-[13px] font-extrabold text-white",
            "bg-[linear-gradient(135deg,#FF5A1F_0%,#FF8A3D_100%)]",
            "shadow-[0_0_0_1px_rgba(255,255,255,0.08),inset_0_1px_0_rgba(255,255,255,0.25)]",
          )}
        >
          A
        </div>
        <span className="font-bold">Agent Arena</span>
        <span className="ml-1 text-xs font-normal text-arena-text-mute-dark">
          / PAI 控制台
        </span>
      </div>

      <div className="mx-4 h-[18px] w-px bg-[#2A323F]" />

      <nav className="hidden items-center gap-2 text-xs text-arena-text-mute-dark sm:flex">
        <span>智能体平台</span>
        <span className="text-[#404a59]">/</span>
        <span>评测中心</span>
        <span className="text-[#404a59]">/</span>
        <b className="font-semibold text-white">{PAGE_LABEL[view]}</b>
      </nav>

      <div className="ml-auto flex items-center gap-1.5">
        <TopBarPill kind="ok">
          <span className="size-1.5 rounded-full bg-[#00D17C] shadow-[0_0_6px_#00d17c80]" />
          <span>cn-hangzhou</span>
        </TopBarPill>
        <TopBarPill kind={configured ? "ok" : "warn"}>
          <span
            className={cn(
              "size-1.5 rounded-full",
              configured
                ? "bg-[#00D17C] shadow-[0_0_6px_#00d17c80]"
                : "bg-arena-warning shadow-[0_0_6px_rgba(232,156,26,0.6)]",
            )}
          />
          <span>{agentCount ?? 2} Agents {configured ? "online" : "未配置"}</span>
        </TopBarPill>
        <TopBarIcon title="文档">
          <BookOpen className="size-4" />
        </TopBarIcon>
        <TopBarIcon title="通知">
          <Bell className="size-4" />
        </TopBarIcon>
        <button
          type="button"
          onClick={onAccessKey}
          title="访问密钥"
          className="ml-1 flex items-center gap-2 rounded-full bg-[#1A2029] py-1 pl-1 pr-2.5 transition hover:bg-[#222933]"
        >
          <span className="grid size-6 place-items-center rounded-full bg-[linear-gradient(135deg,#FF5A1F,#C73A0E)] font-mono text-[11px] font-bold text-white">
            XW
          </span>
          <span className="text-xs font-medium text-white">xiaowen.l</span>
        </button>
      </div>
    </header>
  )
}

function TopBarPill({
  children,
  kind = "ok",
}: {
  children: React.ReactNode
  kind?: "ok" | "warn"
}) {
  return (
    <div
      className={cn(
        "inline-flex cursor-default items-center gap-1.5 rounded-arena-sm border border-[#252C36] bg-[#1A2029] px-2.5 py-[5px] font-mono text-xs text-[#C2C9D4] transition",
        "hover:bg-[#222933] hover:text-white",
        kind === "warn" && "border-arena-warning/40",
      )}
    >
      {children}
    </div>
  )
}

function TopBarIcon({
  children,
  title,
}: {
  children: React.ReactNode
  title: string
}) {
  return (
    <div
      title={title}
      className="grid size-8 cursor-pointer place-items-center rounded-arena-sm text-[#C2C9D4] transition hover:bg-[#1A2029] hover:text-white"
    >
      {children}
    </div>
  )
}
