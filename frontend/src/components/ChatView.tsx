import { useEffect } from "react";
import { PanelLeft } from "lucide-react";
import { activeRuntime, EMPTY_MESSAGES, useChatStore } from "../store/chat";
import { useConversationsStore } from "../store/conversations";
import { useComposer } from "../store/composer";
import { useResponsesChat } from "../hooks/useResponsesChat";
import { MessageList } from "./MessageList";
import { Composer } from "./Composer";
import { AgentSelector } from "./AgentSelector";
import { BrandMark } from "./Sidebar";
import { ICON_BTN } from "../lib/ui";
import { useI18n } from "../i18n";

export function ChatView({ onToggleSidebar }: { onToggleSidebar?: () => void }) {
  const { t } = useI18n();
  const messages = useChatStore((s) => activeRuntime(s)?.messages ?? EMPTY_MESSAGES);
  const conversations = useConversationsStore((s) => s.items);
  const selectedId = useConversationsStore((s) => s.selectedId);
  const { send, stop, regenerate, isStreaming, resumeIfInterrupted } =
    useResponsesChat();

  // Expose send() globally so the inline HITL card/dialog can resume a paused
  // turn (send a "继续" message) after the user authorizes.
  useEffect(() => {
    useComposer.getState().setSubmit(send);
    return () => useComposer.getState().setSubmit(null);
  }, [send]);

  const currentTitle = (() => {
    if (!selectedId) return null;
    const c = conversations.find((it) => it.id === selectedId);
    return c?.title || null;
  })();

  useEffect(() => {
    const tryResume = () => {
      if (document.visibilityState === "visible") void resumeIfInterrupted();
    };
    tryResume();
    document.addEventListener("visibilitychange", tryResume);
    window.addEventListener("online", tryResume);
    return () => {
      document.removeEventListener("visibilitychange", tryResume);
      window.removeEventListener("online", tryResume);
    };
  }, [resumeIfInterrupted]);

  return (
    <div className="flex h-full flex-col">
      {/* Top bar */}
      <div className="h-[var(--header-h)] flex items-center gap-1 border-b border-[var(--border)] bg-[var(--bg-elevated)]/92 px-4 flex-shrink-0 shadow-[0_1px_0_rgba(15,23,42,0.02)] backdrop-blur">
        {onToggleSidebar && (
          <button
            type="button"
            aria-label={t("chat.toggleSidebar")}
            title={t("chat.toggleSidebar")}
            onClick={onToggleSidebar}
            className={ICON_BTN}
          >
            <PanelLeft className="h-4 w-4" />
          </button>
        )}
        <div className="flex-1 min-w-0 text-center px-2">
          {currentTitle && (
            <span className="truncate inline-block max-w-full rounded-full bg-[var(--surface)] px-3 py-1 text-xs font-medium text-[var(--text-muted)]">
              {currentTitle}
            </span>
          )}
        </div>
        <AgentSelector />
      </div>

      {/* Main area: empty state or messages + composer */}
      {messages.length === 0 ? (
        <div className="flex-1 grid place-items-center px-4 pb-12">
          <div className="chat-container flex flex-col items-center gap-5">
            <BrandMark size="lg" />
            <h1 className="text-xl font-semibold text-[var(--text)] tracking-tight">
              {t("chat.greeting")}
            </h1>
            <div className="w-full max-w-[720px]">
              <Composer onSend={send} onStop={stop} isStreaming={isStreaming} />
            </div>
          </div>
        </div>
      ) : (
        <>
          <MessageList onRegenerate={regenerate} />
          <div
            className="flex-shrink-0 border-t border-[var(--border)] bg-[var(--bg-elevated)]/92 py-3 backdrop-blur"
            style={{ boxShadow: "0 -10px 28px -26px rgba(15,23,42,0.45)" }}
          >
            <Composer onSend={send} onStop={stop} isStreaming={isStreaming} />
          </div>
        </>
      )}
    </div>
  );
}
