import { useEffect } from "react";
import { PanelLeft } from "lucide-react";
import { useChatStore } from "../store/chat";
import { useResponsesChat } from "../hooks/useResponsesChat";
import { MessageList } from "./MessageList";
import { Composer } from "./Composer";
import { ModelSelector } from "./ModelSelector";
import { ThemeToggle } from "./ThemeToggle";
import { BrandMark } from "./Sidebar";

export function ChatView({ onToggleSidebar }: { onToggleSidebar?: () => void }) {
  const model = useChatStore((s) => s.model);
  const setModel = useChatStore((s) => s.setModel);
  const messages = useChatStore((s) => s.messages);
  const { send, stop, regenerate, isStreaming, resumeIfInterrupted } =
    useResponsesChat();

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
      {/* Minimal top bar */}
      <div className="h-12 flex items-center gap-1 px-3 border-b border-[var(--border)] flex-shrink-0 bg-[var(--bg)]">
        {onToggleSidebar && (
          <button
            type="button"
            aria-label="Toggle sidebar"
            onClick={onToggleSidebar}
            className="rounded-[var(--radius-sm)] p-1.5 text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] transition-colors"
          >
            <PanelLeft className="h-4 w-4" />
          </button>
        )}
        <div className="flex-1" />
        <ModelSelector model={model} onChange={setModel} />
        <ThemeToggle />
      </div>

      {/* Main area: empty state or messages + composer */}
      {messages.length === 0 ? (
        <div className="flex-1 grid place-items-center px-4 pb-8">
          <div className="flex flex-col items-center gap-5 w-full max-w-2xl">
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
          <div className="flex-shrink-0 border-t border-[var(--border)] bg-[var(--bg)] py-3">
            <Composer onSend={send} onStop={stop} isStreaming={isStreaming} />
          </div>
        </>
      )}
    </div>
  );
}
