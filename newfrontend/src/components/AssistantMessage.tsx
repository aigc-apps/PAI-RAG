import type { ChatMessage } from "../types";
import { Markdown } from "./Markdown";
import { CollapsibleReasoning } from "./CollapsibleReasoning";
import { MessageControls } from "./MessageControls";

export function AssistantMessage({
  message,
  onRegenerate,
}: {
  message: ChatMessage;
  onRegenerate?: () => void;
}) {
  return (
    <div className="flex flex-col items-start">
      <div className="max-w-[80%]">
        <CollapsibleReasoning
          reasoning={message.reasoning}
          status={message.reasoningStatus}
        />
        {message.status === "failed" ? (
          <div className="rounded-md bg-red-50 px-3 py-2 text-red-700">
            {message.error || "Something went wrong."}
          </div>
        ) : (
          <Markdown content={message.text} />
        )}
        {message.status === "stopped" && (
          <div className="mt-1 text-xs italic text-gray-400">stopped</div>
        )}
        {message.status === "completed" && (
          <MessageControls text={message.text} onRegenerate={onRegenerate} />
        )}
      </div>
    </div>
  );
}
