import { useEffect, useRef } from "react";
import { useChatStore } from "../store/chat";
import { UserMessage } from "./UserMessage";
import { AssistantMessage } from "./AssistantMessage";

export function MessageList({ onRegenerate }: { onRegenerate: () => void }) {
  const messages = useChatStore((s) => s.messages);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const lastAssistantIndex = messages
    .map((m) => m.role)
    .lastIndexOf("assistant");

  return (
    <div className="flex-1 space-y-4 overflow-y-auto p-4">
      {messages.map((m, i) =>
        m.role === "user" ? (
          <UserMessage key={m.id} message={m} />
        ) : (
          <AssistantMessage
            key={m.id}
            message={m}
            onRegenerate={i === lastAssistantIndex ? onRegenerate : undefined}
          />
        )
      )}
      <div ref={bottomRef} />
    </div>
  );
}
