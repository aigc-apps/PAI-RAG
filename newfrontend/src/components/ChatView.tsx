import { useEffect } from "react";
import { PanelLeft } from "lucide-react";
import { useChatStore } from "../store/chat";
import { useResponsesChat } from "../hooks/useResponsesChat";
import { MessageList } from "./MessageList";
import { Composer } from "./Composer";
import { ModelSelector } from "./ModelSelector";
import { ThemeToggle } from "./ThemeToggle";

export function ChatView({ onToggleSidebar }: { onToggleSidebar?: () => void }) {
  const model = useChatStore((s) => s.model);
  const setModel = useChatStore((s) => s.setModel);
  const messages = useChatStore((s) => s.messages);
  const { send, stop, regenerate, isStreaming, resumeIfInterrupted } =
    useResponsesChat();

  // Resume an interrupted in-flight answer when the tab/network comes back.
  useEffect(() => {
    const tryResume = () => {
      if (document.visibilityState === "visible") void resumeIfInterrupted();
    };
    tryResume(); // also on mount (no-op unless a streaming message exists)
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
      <div className="h-12 border-b border-[var(--border)] flex items-center px-3 gap-2 flex-shrink-0">
        {onToggleSidebar && (
          <button
            type="button"
            aria-label="Toggle sidebar"
            onClick={onToggleSidebar}
            className="rounded-lg p-1.5 text-[var(--text-muted)] hover:bg-[var(--user-bubble)]"
          >
            <PanelLeft className="h-5 w-5" />
          </button>
        )}
        <ModelSelector model={model} onChange={setModel} />
        <div className="flex-1" />
        <ThemeToggle />
      </div>

      {/* Main area: empty state or messages + composer */}
      {messages.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center px-4 pb-8">
          <h1 className="mb-6 text-2xl font-semibold text-[var(--text)]">
            What can I help with?
          </h1>
          <div className="w-full">
            <Composer onSend={send} onStop={stop} isStreaming={isStreaming} />
          </div>
        </div>
      ) : (
        <>
          <MessageList onRegenerate={regenerate} />
          <div className="flex-shrink-0 py-3">
            <Composer onSend={send} onStop={stop} isStreaming={isStreaming} />
          </div>
        </>
      )}
    </div>
  );
}
