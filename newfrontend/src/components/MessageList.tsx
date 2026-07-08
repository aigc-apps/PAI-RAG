import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import { useChatStore } from "../store/chat";
import { UserMessage } from "./UserMessage";
import { AssistantMessage } from "./AssistantMessage";

// How close to the bottom (px) still counts as "pinned" and keeps auto-following.
const NEAR_BOTTOM_PX = 80;

export function MessageList({ onRegenerate }: { onRegenerate: () => void }) {
  const messages = useChatStore((s) => s.messages);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  // Follow new content only while the user is pinned to the bottom. Set false
  // the moment they scroll up, re-armed when they scroll back down. A ref (not
  // state) so streaming patches read the latest value without re-rendering.
  const stickRef = useRef(true);
  const prevLenRef = useRef(0);
  const [showJump, setShowJump] = useState(false);

  const onScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    const atBottom = distance < NEAR_BOTTOM_PX;
    stickRef.current = atBottom;
    setShowJump(!atBottom);
  }, []);

  useEffect(() => {
    // A new message (user send or a new assistant turn) re-pins to the bottom;
    // token-by-token growth of the current message only follows while pinned.
    const grew = messages.length > prevLenRef.current;
    prevLenRef.current = messages.length;
    if (grew) stickRef.current = true;
    if (stickRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: "auto" });
      setShowJump(false);
    }
  }, [messages]);

  const jumpToBottom = () => {
    stickRef.current = true;
    setShowJump(false);
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const lastAssistantIndex = messages
    .map((m) => m.role)
    .lastIndexOf("assistant");

  return (
    <div className="relative flex-1 min-h-0">
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="h-full overflow-y-auto scrollbar-thin"
        style={{ scrollbarGutter: "stable" }}
      >
        <div className="chat-container px-6 py-8 space-y-6">
          {messages.map((m, i) =>
            m.role === "user" ? (
              <UserMessage key={m.id} message={m} />
            ) : (
              <AssistantMessage
                key={m.id}
                message={m}
                isLast={i === lastAssistantIndex}
                onRegenerate={i === lastAssistantIndex ? onRegenerate : undefined}
              />
            )
          )}
          <div ref={bottomRef} />
        </div>
      </div>
      {showJump && (
        <button
          type="button"
          onClick={jumpToBottom}
          aria-label="回到底部"
          className="absolute bottom-4 left-1/2 -translate-x-1/2 inline-flex items-center gap-1.5 rounded-full border border-[var(--border)] bg-[var(--surface)] px-3 py-1.5 text-xs font-medium text-[var(--text-muted)] shadow-md transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
        >
          <ChevronDown className="h-3.5 w-3.5" />
          回到底部
        </button>
      )}
    </div>
  );
}
