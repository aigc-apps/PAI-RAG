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
    <div className="flex-1 overflow-y-auto scrollbar-thin" style={{ scrollbarGutter: "stable" }}>
      <div className="chat-container px-6 py-8 space-y-6">
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
    </div>
  );
}
