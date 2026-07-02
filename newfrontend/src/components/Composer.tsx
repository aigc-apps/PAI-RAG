import { useRef, useState, type KeyboardEvent } from "react";
import { ArrowUp, Square } from "lucide-react";

export function Composer({
  onSend,
  onStop,
  isStreaming,
}: {
  onSend: (text: string) => void;
  onStop: () => void;
  isStreaming: boolean;
}) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const autoGrow = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 144)}px`;
  };

  const submit = () => {
    const text = value.trim();
    if (!text) return;
    onSend(text);
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isStreaming) submit();
    }
  };

  return (
    <div className="chat-container px-4">
      <div className="flex items-end gap-2 border border-[var(--border)] rounded-[var(--radius-lg)] bg-[var(--bg-elevated)] px-3 py-2 focus-within:border-[var(--border-strong)] transition-colors">
        <textarea
          ref={textareaRef}
          className="flex-1 resize-none border-0 outline-none bg-transparent text-[var(--text)] placeholder:text-[var(--text-faint)] py-0.5 leading-6 text-sm"
          placeholder="Send a message…"
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            autoGrow();
          }}
          onKeyDown={onKeyDown}
          rows={1}
        />
        {isStreaming ? (
          <button
            type="button"
            aria-label="Stop"
            onClick={onStop}
            className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-[var(--radius-sm)] border border-[var(--border-strong)] text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] transition-colors"
          >
            <Square className="h-3 w-3" />
          </button>
        ) : (
          <button
            type="button"
            aria-label="Send"
            onClick={submit}
            className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-[var(--radius-sm)] bg-[var(--surface-3)] text-[var(--text)] hover:bg-[var(--border-strong)] disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            disabled={!value.trim()}
          >
            <ArrowUp className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  );
}
