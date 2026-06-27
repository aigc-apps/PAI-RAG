import type { ChatMessage } from "../types";

export function UserMessage({ message }: { message: ChatMessage }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] whitespace-pre-wrap rounded-3xl bg-[var(--user-bubble)] px-4 py-2.5 text-[var(--text)]">
        {message.text}
      </div>
    </div>
  );
}
