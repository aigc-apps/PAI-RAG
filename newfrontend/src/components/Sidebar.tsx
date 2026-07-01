import { useEffect } from "react";
import { Plus, Settings2, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { useConversationsStore } from "../store/conversations";
import { useChatStore } from "../store/chat";
import { getConversation } from "../api/conversations";
import { cn } from "../lib/cn";

export function BrandMark({ size = "sm" }: { size?: "sm" | "lg" }) {
  const dotCls = size === "lg" ? "h-7 w-7" : "h-5 w-5";
  const textCls = size === "lg" ? "text-xl" : "text-sm";
  return (
    <div className="flex items-center gap-2">
      <span
        className={`${dotCls} rounded-[var(--radius-sm)] flex-shrink-0 bg-[var(--surface-3)]`}
      />
      <span className={`${textCls} font-semibold tracking-tight text-[var(--text)]`}>
        Aria
      </span>
    </div>
  );
}

export function Sidebar({ onOpenSettings }: { onOpenSettings?: () => void }) {
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
        <div className="flex-1" />
        {onOpenSettings && (
          <button
            type="button"
            aria-label="Open settings"
            onClick={onOpenSettings}
            className="rounded-[var(--radius-sm)] p-1.5 text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] transition-colors"
          >
            <Settings2 className="h-4 w-4" />
          </button>
        )}
      </div>

      <div className="px-3 pt-3">
        <button
          type="button"
          aria-label="New chat"
          onClick={newChat}
          className="flex w-full items-center gap-2 rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-2 text-sm font-medium text-[var(--text)] bg-[var(--bg)] hover:bg-[var(--surface-2)] transition-colors"
        >
          <Plus className="h-4 w-4 flex-shrink-0 text-[var(--text-muted)]" />
          New chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin px-2 pt-2 pb-2">
        {items.length === 0 ? (
          <div className="px-2 py-4 text-xs text-[var(--text-faint)]">
            No conversations yet
          </div>
        ) : (
          items.map((c) => (
            <div
              key={c.id}
              onClick={() => openConversation(c.id)}
              className={cn(
                "group flex cursor-pointer items-center justify-between rounded-[var(--radius-sm)] px-3 py-2 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] transition-colors",
                selectedId === c.id &&
                  "bg-[var(--surface-2)] text-[var(--text)] font-medium"
              )}
            >
              <span className="truncate">{c.title || "Untitled"}</span>
              <button
                type="button"
                aria-label="Delete conversation"
                onClick={(e) => onDelete(e, c.id)}
                className="invisible flex-shrink-0 text-[var(--text-faint)] group-hover:visible hover:text-[var(--danger)]"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
