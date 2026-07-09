import { useEffect } from "react";
import { Database, Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { useConversationsStore } from "../store/conversations";
import { useChatStore } from "../store/chat";
import { useAgentsStore } from "../store/agents";
import { getConversation } from "../api/conversations";
import { cn } from "../lib/cn";
import { UserMenu } from "./UserMenu";

export function BrandMark({ size = "sm" }: { size?: "sm" | "lg" }) {
  const dotCls = size === "lg" ? "h-7 w-7" : "h-5 w-5";
  const textCls = size === "lg" ? "text-xl" : "text-sm";
  const agents = useAgentsStore((s) => s.agents);
  const defaultAgent = useAgentsStore((s) => s.defaultAgent);
  const agentId = useChatStore((s) => s.agentId);
  const name =
    agents.find((a) => a.id === (agentId || defaultAgent))?.name || "MiniAgent";
  return (
    <div className="flex items-center gap-2">
      <span
        className={`${dotCls} rounded-[var(--radius-sm)] flex-shrink-0 bg-[var(--surface-3)]`}
      />
      <span className={`${textCls} font-semibold tracking-tight text-[var(--text)]`}>
        {name}
      </span>
    </div>
  );
}

export function Sidebar({
  onOpenSettings,
  onOpenKnowledge,
}: {
  onOpenSettings?: () => void;
  onOpenKnowledge?: () => void;
}) {
  const items = useConversationsStore((s) => s.items);
  const selectedId = useConversationsStore((s) => s.selectedId);
  const refresh = useConversationsStore((s) => s.refresh);
  const select = useConversationsStore((s) => s.select);
  const clearSelection = useConversationsStore((s) => s.clearSelection);
  const remove = useConversationsStore((s) => s.remove);
  const loadHistory = useChatStore((s) => s.loadHistory);
  const reset = useChatStore((s) => s.reset);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const openConversation = async (id: string) => {
    try {
      const detail = await getConversation(id);
      loadHistory(detail);
      select(id);
    } catch {
      toast.error("Could not load conversation");
      clearSelection();
    }
  };

  const newChat = () => {
    reset();
    clearSelection();
  };

  const onDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await remove(id);
      if (selectedId === id) newChat();
    } catch {
      toast.error("Could not delete conversation");
    }
  };

  return (
    <aside className="flex w-[240px] flex-col border-r border-[var(--border)] bg-[var(--surface)] h-full flex-shrink-0">
      <div className="flex items-center gap-2 px-4 h-12 border-b border-[var(--border)] flex-shrink-0">
        <BrandMark />
      </div>

      <div className="px-3 pt-3">
        <button
          type="button"
          aria-label="New chat"
          onClick={newChat}
          className="flex w-full items-center justify-center gap-2 rounded-[var(--radius-sm)] border border-[var(--border-strong)] bg-[var(--bg)] px-3 py-2 text-sm font-medium text-[var(--text)] shadow-[var(--shadow-sm)] hover:bg-[var(--surface-2)] hover:border-[var(--accent)]/40 transition-colors focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[var(--accent)]"
        >
          <Plus className="h-4 w-4 flex-shrink-0 text-[var(--text-muted)]" />
          新建对话
        </button>
        {onOpenKnowledge && (
          <button
            type="button"
            aria-label="Open knowledge"
            onClick={onOpenKnowledge}
            className="mt-2 flex w-full items-center justify-center gap-2 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm font-medium text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] transition-colors focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[var(--accent)]"
          >
            <Database className="h-4 w-4 flex-shrink-0" />
            知识库
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin px-2 pt-2 pb-2">
        {items.length === 0 ? (
          <div className="px-2 py-4 text-xs text-[var(--text-faint)]">
            暂无对话
          </div>
        ) : (
          items.map((c) => {
            const selected = selectedId === c.id;
            return (
              <div
                key={c.id}
                onClick={() => openConversation(c.id)}
                title={c.title || "Untitled"}
                className={cn(
                  "group relative flex cursor-pointer items-center justify-between rounded-[var(--radius-sm)] px-3 py-2 text-sm transition-colors",
                  selected
                    ? "bg-[var(--surface-2)] text-[var(--text)] font-medium"
                    : "text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
                )}
              >
                {selected && (
                  <span
                    className="absolute left-0 top-1/2 h-5 -translate-y-1/2 w-[3px] rounded-r bg-[var(--accent)]"
                    aria-hidden="true"
                  />
                )}
                <span className="truncate">{c.title || "Untitled"}</span>
                <button
                  type="button"
                  aria-label="删除对话"
                  title="删除对话"
                  onClick={(e) => onDelete(e, c.id)}
                  className="invisible flex-shrink-0 rounded p-1 text-[var(--text-faint)] group-hover:visible hover:text-[var(--danger)] hover:bg-[var(--danger)]/10 focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[var(--accent)]"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            );
          })
        )}
      </div>

      {/* Account + settings, pinned to the bottom-left */}
      <div className="border-t border-[var(--border)] p-2 flex-shrink-0">
        <UserMenu onOpenSettings={onOpenSettings} />
      </div>
    </aside>
  );
}
