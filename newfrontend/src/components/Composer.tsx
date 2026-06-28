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
    // max ~6 lines at ~24px per line
    el.style.height = `${Math.min(el.scrollHeight, 144)}px`;
  };

  const submit = () => {
    const text = value.trim();
    if (!text) return;
    onSend(text);
    setValue("");
    // reset height
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
    <div className="w-full max-w-3xl mx-auto px-4">
      <div className="bg-[var(--bg-elevated)] border border-[var(--border-strong)] rounded-[var(--radius-lg)] shadow-[var(--shadow)] px-3 py-2 flex items-end gap-2 focus-within:border-[var(--accent)] transition-colors">
        <textarea
          ref={textareaRef}
          className="flex-1 resize-none border-0 outline-none bg-transparent text-[var(--text)] placeholder:text-[var(--text-faint)] py-0.5 leading-6"
          placeholder="Message Aria…"
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
            className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded text-[var(--accent-fg)] hover:opacity-90"
            style={{ background: "var(--accent-grad)" }}
          >
            <Square className="h-4 w-4" />
          </button>
        ) : (
          <button
            type="button"
            aria-label="Send"
            onClick={submit}
            className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full text-[var(--accent-fg)] hover:opacity-90 disabled:opacity-40"
            style={{ background: "var(--accent-grad)" }}
            disabled={!value.trim()}
          >
            <ArrowUp className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  );
}
