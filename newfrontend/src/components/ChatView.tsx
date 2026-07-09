import { useEffect } from "react";
import { PanelLeft } from "lucide-react";
import { useChatStore } from "../store/chat";
import { useConversationsStore } from "../store/conversations";
import { useComposer } from "../store/composer";
import { useResponsesChat } from "../hooks/useResponsesChat";
import { MessageList } from "./MessageList";
import { Composer } from "./Composer";
import { ModelSelector } from "./ModelSelector";
import { AgentSelector } from "./AgentSelector";
import { ThemeToggle } from "./ThemeToggle";
import { BrandMark } from "./Sidebar";

export function ChatView({ onToggleSidebar }: { onToggleSidebar?: () => void }) {
  const model = useChatStore((s) => s.model);
  const setModel = useChatStore((s) => s.setModel);
  const messages = useChatStore((s) => s.messages);
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
      <div className="h-12 flex items-center gap-1 px-3 border-b border-[var(--border)] flex-shrink-0 bg-[var(--bg)]">
        {onToggleSidebar && (
          <button
            type="button"
            aria-label="Toggle sidebar"
            title="切换侧边栏"
            onClick={onToggleSidebar}
            className="icon-btn p-1.5 text-[var(--text-muted)] hover:text-[var(--text)]"
          >
            <PanelLeft className="h-4 w-4" />
          </button>
        )}
        <div className="flex-1 min-w-0 text-center px-2">
          {currentTitle && (
            <span className="truncate inline-block max-w-full text-sm font-medium text-[var(--text-muted)]">
              {currentTitle}
            </span>
          )}
        </div>
        <AgentSelector />
        <ModelSelector model={model} onChange={setModel} />
        <ThemeToggle />
      </div>

      {/* Main area: empty state or messages + composer */}
      {messages.length === 0 ? (
        <div className="flex-1 grid place-items-center px-4 pb-8">
          <div className="chat-container flex flex-col items-center gap-5">
            <BrandMark size="lg" />
            <h1 className="text-xl font-semibold text-[var(--text)] tracking-tight">
              How can I help today?
            </h1>
            <div className="w-full">
              <Composer onSend={send} onStop={stop} isStreaming={isStreaming} />
            </div>
          </div>
        </div>
      ) : (
        <>
          <MessageList onRegenerate={regenerate} />
          <div
            className="flex-shrink-0 border-t border-[var(--border)] bg-[var(--bg)] py-3"
            style={{ boxShadow: "0 -1px 8px rgba(0,0,0,0.04)" }}
          >
            <Composer onSend={send} onStop={stop} isStreaming={isStreaming} />
          </div>
        </>
      )}
    </div>
  );
}
