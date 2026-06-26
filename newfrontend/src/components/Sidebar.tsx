import { useEffect } from "react";
import { Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { useConversationsStore } from "../store/conversations";
import { useChatStore } from "../store/chat";
import { getConversation } from "../api/conversations";
import { cn } from "../lib/cn";

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
    <aside className="flex w-64 flex-col border-r border-gray-200 bg-gray-50">
      <button
        type="button"
        onClick={newChat}
        className="m-2 flex items-center gap-2 rounded-md bg-blue-600 px-3 py-2 text-white"
      >
        <Plus className="h-4 w-4" /> New chat
      </button>
      <div className="flex-1 overflow-y-auto">
        {items.map((c) => (
          <div
            key={c.id}
            onClick={() => openConversation(c.id)}
            className={cn(
              "group flex cursor-pointer items-center justify-between px-3 py-2 text-sm hover:bg-gray-100",
              selectedId === c.id && "bg-gray-200"
            )}
          >
            <span className="truncate">{c.title || "Untitled"}</span>
            <button
              type="button"
              aria-label="Delete conversation"
              onClick={(e) => onDelete(e, c.id)}
              className="invisible text-gray-400 group-hover:visible hover:text-red-600"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
