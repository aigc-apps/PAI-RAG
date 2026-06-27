import type { ChatMessage } from "../types";
import { Markdown } from "./Markdown";
import { CollapsibleReasoning } from "./CollapsibleReasoning";
import { MessageControls } from "./MessageControls";
import { ToolCall } from "./ToolCall";

export function AssistantMessage({
  message,
  onRegenerate,
}: {
  message: ChatMessage;
  onRegenerate?: () => void;
}) {
  const showControls =
    message.status === "completed" || message.status === "cancelled";
  return (
    <div className="flex w-full flex-col items-start">
      <CollapsibleReasoning
        reasoning={message.reasoning}
        status={message.reasoningStatus}
      />
      {message.toolCalls.map((t) => (
        <ToolCall key={t.id} tool={t} />
      ))}
      {message.status === "failed" ? (
        <div className="rounded-md bg-red-50 px-3 py-2 text-red-700">
          {message.error || "Something went wrong."}
        </div>
      ) : (
        message.text && <Markdown content={message.text} />
      )}
      {message.status === "stopped" && (
        <div className="mt-1 text-xs italic text-[var(--text-faint)]">stopped</div>
      )}
      {message.status === "cancelled" && (
        <div className="mt-1 text-xs italic text-[var(--text-faint)]">cancelled</div>
      )}
      {showControls && (
        <MessageControls text={message.text} onRegenerate={onRegenerate} />
      )}
    </div>
  );
}
