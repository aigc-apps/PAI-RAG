import { useEffect } from "react";
import { Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { useConversationsStore } from "../store/conversations";
import { useChatStore, type ConvRuntime } from "../store/chat";
import { getConversation } from "../api/conversations";
import { cn } from "../lib/cn";
import { UserMenu } from "./UserMenu";
import { useI18n, translate, useI18nStore } from "../i18n";

/** A conversation's title while it lives only in a local runtime (before the
 * server list has it): the first user turn, or a placeholder for an empty draft. */
function runtimeTitle(rt: ConvRuntime): string {
  const firstUser = rt.messages.find((m) => m.role === "user" && m.text.trim());
  if (firstUser) return firstUser.text.trim();
  return translate(useI18nStore.getState().lang, "chat.newConversationTitle");
}

/** True while the conversation's tail assistant message is still streaming —
 * including a backgrounded run (local loop idle, message still open). */
function isBusy(rt: ConvRuntime): boolean {
  const last = rt.messages[rt.messages.length - 1];
  return last?.role === "assistant" && last.status === "streaming";
}

export function BrandMark({ size = "sm" }: { size?: "sm" | "lg" }) {
  const dotCls = size === "lg" ? "h-9 w-9" : "h-7 w-7";
  const textCls = size === "lg" ? "text-xl" : "text-[15px]";
  const loopCls = size === "lg" ? "text-[10px]" : "text-[8px]";
  return (
    <div className="flex items-center gap-2">
      <span
        className={`${dotCls} grid flex-shrink-0 place-items-center rounded-[var(--radius)] border border-[var(--border)] bg-gradient-to-br from-[var(--bg-elevated)] to-[var(--surface-2)] font-semibold tracking-tight text-[var(--text)] shadow-[var(--shadow-sm)]`}
      >
        <span className={loopCls}>PAI</span>
      </span>
      <span className={`${textCls} font-semibold tracking-tight text-[var(--text)]`}>
        PAI-Loop
      </span>
    </div>
  );
}

export function Sidebar({
  onOpenSettings,
  onOpenUsers,
}: {
  onOpenSettings?: () => void;
  onOpenUsers?: () => void;
}) {
  const { t } = useI18n();
  const items = useConversationsStore((s) => s.items);
  const refresh = useConversationsStore((s) => s.refresh);
  const select = useConversationsStore((s) => s.select);
  const clearSelection = useConversationsStore((s) => s.clearSelection);
  const remove = useConversationsStore((s) => s.remove);
  const runtimes = useChatStore((s) => s.runtimes);
  const activeKey = useChatStore((s) => s.activeKey);
  const activate = useChatStore((s) => s.activate);
  const activateByConversationId = useChatStore((s) => s.activateByConversationId);
  const hydrate = useChatStore((s) => s.hydrate);
  const newDraft = useChatStore((s) => s.newDraft);
  const dropByConversationId = useChatStore((s) => s.dropByConversationId);
  const dropByKey = useChatStore((s) => s.dropByKey);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Highlight and streaming state key off the on-screen runtime, not the
  // server-side `selectedId` — a brand-new/streaming conversation isn't in the
  // server list yet, so `selectedId` alone can't represent or highlight it.
  const activeConversationId = runtimes[activeKey]?.conversationId;
  const serverIds = new Set(items.map((i) => i.id));
  const busyIds = new Set(
    Object.values(runtimes)
      .filter((rt) => rt.conversationId && isBusy(rt))
      .map((rt) => rt.conversationId as string)
  );
  // Local runtimes not yet reflected in the server list: the active draft (shown
  // the moment "新建对话" is clicked, even empty) and any conversation still
  // streaming before its completion refresh lands. Newest first.
  const overlay = Object.values(runtimes)
    .filter((rt) => !rt.conversationId || !serverIds.has(rt.conversationId))
    .filter((rt) => rt.key === activeKey || rt.messages.length > 0)
    .reverse();

  const openRuntime = (rt: ConvRuntime) => {
    activate(rt.key);
    if (rt.conversationId) select(rt.conversationId);
    else clearSelection();
  };

  // Delete an overlay row. An unsent draft (no conversationId) is discarded
  // locally; a persisted/persisting one deletes server-side too.
  const onDeleteRuntime = async (e: React.MouseEvent, rt: ConvRuntime) => {
    e.stopPropagation();
    if (!rt.conversationId) {
      dropByKey(rt.key);
      // Sync the sidebar highlight to whatever runtime became active.
      const next = useChatStore.getState();
      const cid = next.runtimes[next.activeKey]?.conversationId;
      if (cid) select(cid);
      else clearSelection();
      return;
    }
    try {
      await remove(rt.conversationId);
      dropByConversationId(rt.conversationId);
    } catch {
      toast.error(t("sidebar.deleteFailed"));
    }
  };

  const openConversation = async (id: string) => {
    // Prefer an already-loaded runtime (it may be mid-stream) so its partial
    // tokens aren't clobbered by staler server history; only fetch the first
    // time we open a conversation this session.
    if (activateByConversationId(id)) {
      select(id);
      return;
    }
    try {
      const detail = await getConversation(id);
      hydrate(detail);
      select(id);
    } catch {
      toast.error(t("sidebar.loadFailed"));
      clearSelection();
    }
  };

  const newChat = () => {
    newDraft();
    clearSelection();
  };

  const onDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await remove(id);
      // Drops the conversation's runtime; if it was on screen, this installs a
      // fresh draft (an empty "new chat" view). `remove` already clears the
      // sidebar selection when the deleted item was selected.
      dropByConversationId(id);
    } catch {
      toast.error(t("sidebar.deleteFailed"));
    }
  };

  return (
    <aside className="flex w-[var(--sidebar-w)] flex-col border-r border-[var(--border)] bg-[var(--surface)]/95 h-full flex-shrink-0 shadow-[1px_0_0_rgba(15,23,42,0.02)]">
      <div className="flex h-[var(--header-h)] flex-shrink-0 items-center gap-2 border-b border-[var(--border)] bg-[var(--bg-elevated)]/70 px-4">
        <BrandMark />
      </div>

      <div className="px-3 pt-4">
        <button
          type="button"
          aria-label={t("sidebar.newChat")}
          onClick={newChat}
          className="focus-ring flex h-9 w-full items-center justify-center gap-2 rounded-[var(--radius)] border border-[var(--border-strong)] bg-[var(--bg-elevated)] px-3 text-sm font-semibold text-[var(--text)] shadow-[var(--shadow-sm)] transition-colors hover:border-[var(--text-muted)] hover:bg-[var(--surface)]"
        >
          <Plus className="h-4 w-4 flex-shrink-0 text-[var(--text-muted)]" />
          {t("sidebar.newChat")}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin px-2.5 pt-3 pb-3">
        {overlay.length === 0 && items.length === 0 ? (
          <div className="px-2 py-4 text-xs text-[var(--text-faint)]">
            {t("sidebar.empty")}
          </div>
        ) : (
          <>
            {/* Local, not-yet-persisted conversations (new draft + in-flight
                streams) render first so a new chat is switchable immediately. */}
            {overlay.map((rt) => (
              <ConversationRow
                key={rt.key}
                title={runtimeTitle(rt)}
                placeholder={rt.messages.length === 0}
                selected={rt.key === activeKey}
                busy={isBusy(rt)}
                onOpen={() => openRuntime(rt)}
                onDelete={(e) => onDeleteRuntime(e, rt)}
              />
            ))}
            {items.map((c) => (
              <ConversationRow
                key={c.id}
                title={c.title || t("sidebar.untitled")}
                selected={c.id === activeConversationId}
                busy={busyIds.has(c.id)}
                onOpen={() => openConversation(c.id)}
                onDelete={(e) => onDelete(e, c.id)}
              />
            ))}
          </>
        )}
      </div>

      {/* Account + settings, pinned to the bottom-left */}
      <div className="border-t border-[var(--border)] bg-[var(--bg-elevated)]/65 p-2.5 flex-shrink-0">
        <UserMenu onOpenSettings={onOpenSettings} onOpenUsers={onOpenUsers} />
      </div>
    </aside>
  );
}

function ConversationRow({
  title,
  selected,
  busy,
  placeholder,
  onOpen,
  onDelete,
}: {
  title: string;
  selected: boolean;
  busy: boolean;
  placeholder?: boolean;
  onOpen: () => void;
  onDelete?: (e: React.MouseEvent) => void;
}) {
  const { t } = useI18n();
  return (
    <div
      onClick={onOpen}
      title={title}
      className={cn(
        "group relative flex min-h-9 cursor-pointer items-center justify-between gap-2 rounded-[var(--radius)] px-3 py-1.5 text-sm transition-colors",
        selected
          ? "bg-[var(--bg-elevated)] text-[var(--text)] font-semibold shadow-[var(--shadow-sm)]"
          : "text-[var(--text-muted)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text)]"
      )}
    >
      {selected && (
        <span
          className="absolute left-0 top-1/2 h-5 -translate-y-1/2 w-[3px] rounded-r bg-[var(--accent)]"
          aria-hidden="true"
        />
      )}
      {busy && (
        <span
          className="h-1.5 w-1.5 flex-shrink-0 animate-pulse rounded-full bg-[var(--accent)]"
          aria-label={t("common.generating")}
          title={t("common.generating")}
        />
      )}
      <span className={cn("truncate", placeholder && "text-[var(--text-faint)]")}>
        {title}
      </span>
      {onDelete && (
        <button
          type="button"
          aria-label={t("sidebar.deleteConversation")}
          title={t("sidebar.deleteConversation")}
          onClick={onDelete}
          className="invisible ml-auto flex-shrink-0 rounded p-1 text-[var(--text-faint)] group-hover:visible hover:text-[var(--danger)] hover:bg-[var(--danger)]/10 focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[var(--accent)]"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}
