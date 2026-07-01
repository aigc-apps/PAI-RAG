import type { ChatMessage } from "../types";

export function UserMessage({ message }: { message: ChatMessage }) {
  return (
    <div className="flex justify-end animate-msg-in">
      <div className="max-w-[80%] whitespace-pre-wrap rounded-[var(--radius-lg)] bg-[var(--user-bubble)] px-4 py-2.5 text-sm text-[var(--text)]">
        {message.text}
      </div>
    </div>
  );
}
