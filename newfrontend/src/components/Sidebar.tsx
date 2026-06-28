import { useEffect } from "react";
import { Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { useConversationsStore } from "../store/conversations";
import { useChatStore } from "../store/chat";
import { getConversation } from "../api/conversations";
import { cn } from "../lib/cn";

/** Reusable Aurora brand mark: gradient dot + "Aria" wordmark. */
export function BrandMark({ size = "sm" }: { size?: "sm" | "lg" }) {
  const dotCls = size === "lg" ? "h-9 w-9" : "h-6 w-6";
  const textCls = size === "lg" ? "text-2xl" : "text-lg";
  return (
    <div className="flex items-center gap-2.5">
      <span
        className={`${dotCls} rounded-full flex-shrink-0`}
        style={{ background: "var(--accent-grad)" }}
      />
      <span className={`${textCls} font-semibold text-[var(--text)]`}>Aria</span>
    </div>
  );
}

export function Sidebar() {
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
    <aside className="flex w-[264px] flex-col border-r border-[var(--border)] bg-[var(--surface)] h-full">
      {/* Brand mark header */}
      <div className="flex items-center px-4 h-14 border-b border-[var(--border)] flex-shrink-0">
        <BrandMark />
      </div>

      {/* New chat pill */}
      <div className="p-3">
        <button
          type="button"
          aria-label="New chat"
          onClick={newChat}
          className="flex w-full items-center gap-2 rounded-[var(--radius-sm)] border border-[var(--border-strong)] px-3 py-2 text-sm font-medium text-[var(--text)] hover:bg-[var(--surface-2)] transition-colors"
        >
          <Plus className="h-4 w-4 flex-shrink-0" />
          New chat
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto scrollbar-thin px-2 pb-2">
        {items.map((c) => (
          <div
            key={c.id}
            onClick={() => openConversation(c.id)}
            className={cn(
              "group flex cursor-pointer items-center justify-between rounded-[var(--radius-sm)] px-3 py-2 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]",
              selectedId === c.id &&
                "bg-[var(--accent-soft)] text-[var(--text)] font-medium"
            )}
          >
            <span className="truncate">{c.title || "Untitled"}</span>
            <button
              type="button"
              aria-label="Delete conversation"
              onClick={(e) => onDelete(e, c.id)}
              className="invisible flex-shrink-0 text-[var(--text-faint)] group-hover:visible hover:text-[var(--danger)]"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
