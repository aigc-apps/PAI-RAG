import { useChatStore } from "../store/chat";
import { useResponsesChat } from "../hooks/useResponsesChat";
import { MessageList } from "./MessageList";
import { Composer } from "./Composer";
import { ModelSelector } from "./ModelSelector";

export function ChatView() {
  const model = useChatStore((s) => s.model);
  const setModel = useChatStore((s) => s.setModel);
  const { send, stop, regenerate, isStreaming } = useResponsesChat();

  return (
    <div className="flex h-full flex-1 flex-col">
      <div className="flex items-center justify-between border-b border-gray-200 p-3">
        <span className="font-medium">Agent Chat</span>
        <ModelSelector model={model} onChange={setModel} />
      </div>
      <MessageList onRegenerate={regenerate} />
      <Composer onSend={send} onStop={stop} isStreaming={isStreaming} />
    </div>
  );
}
