import { useState, type KeyboardEvent } from "react";
import { SendHorizontal, Square } from "lucide-react";

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

  const submit = () => {
    const text = value.trim();
    if (!text) return;
    onSend(text);
    setValue("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isStreaming) submit();
    }
  };

  return (
    <div className="flex items-end gap-2 border-t border-gray-200 p-3">
      <textarea
        className="min-h-[44px] flex-1 resize-none rounded-md border border-gray-300 p-2 outline-none focus:border-gray-500"
        placeholder="Send a message…"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKeyDown}
        rows={1}
      />
      {isStreaming ? (
        <button
          type="button"
          aria-label="Stop"
          onClick={onStop}
          className="rounded-md bg-gray-800 p-2 text-white"
        >
          <Square className="h-5 w-5" />
        </button>
      ) : (
        <button
          type="button"
          aria-label="Send"
          onClick={submit}
          className="rounded-md bg-blue-600 p-2 text-white disabled:opacity-50"
          disabled={!value.trim()}
        >
          <SendHorizontal className="h-5 w-5" />
        </button>
      )}
    </div>
  );
}
