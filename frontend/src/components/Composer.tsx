import { useRef, useState, type KeyboardEvent } from "react";
import { ArrowUp, Square } from "lucide-react";
import { useI18n } from "../i18n";

export function Composer({
  onSend,
  onStop,
  isStreaming,
}: {
  onSend: (text: string) => void;
  onStop: () => void;
  isStreaming: boolean;
}) {
  const { t } = useI18n();
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
    // While an IME (e.g. Chinese/Japanese) is composing, Enter confirms the
    // candidate selection — it must not send the message. isComposing covers
    // modern browsers; keyCode 229 is the legacy fallback some IMEs still emit.
    if (e.nativeEvent.isComposing || e.keyCode === 229) return;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isStreaming) submit();
    }
  };

  return (
    <div className="chat-container px-4">
      <div className="professional-card flex items-end gap-2 px-3 py-2 transition-colors focus-within:border-[var(--accent)] focus-within:shadow-[var(--shadow-focus)]">
        <textarea
          ref={textareaRef}
          className="flex-1 resize-none border-0 outline-none bg-transparent text-[var(--text)] placeholder:text-[var(--text-faint)] py-0.5 leading-6 text-sm"
          placeholder={t("composer.placeholder")}
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
            aria-label={t("composer.stop")}
            onClick={onStop}
            className="focus-ring flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-[var(--radius)] border border-[var(--border-strong)] text-[var(--text-muted)] transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
          >
            <Square className="h-3 w-3" />
          </button>
        ) : (
          <button
            type="button"
            aria-label={t("composer.send")}
            onClick={submit}
            className="focus-ring flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-[var(--radius)] border border-[var(--border-strong)] bg-[var(--bg-elevated)] text-[var(--text)] shadow-[var(--shadow-sm)] transition-colors hover:border-[var(--text-muted)] hover:bg-[var(--surface)] disabled:border-[var(--border)] disabled:bg-[var(--surface)] disabled:text-[var(--text-faint)] disabled:opacity-70"
            disabled={!value.trim()}
          >
            <ArrowUp className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  );
}
