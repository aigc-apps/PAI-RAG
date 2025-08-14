import { useState } from "react";
import type { ReasoningContentPartComponent } from "@assistant-ui/react";

export const CollapsibleReasoning: ReasoningContentPartComponent = ({
  text,
  status,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  
  // 直接使用传入的 text，不需要保存到 state
  const displayText = text;

  // 根据 status 判断是否正在流式输出
  const isThinking = status.type === "running";

  // 如果没有内容，不显示
  if (!displayText) return null;

  return (
    <div className="border border-gray-300 rounded-lg p-2 my-2 text-sm bg-gray-50">
      <button
        onClick={() => setIsOpen((prev) => !prev)}
        className="font-semibold text-gray-700 hover:text-gray-900 flex items-center gap-2 w-full text-left"
      >
        <span>{isThinking ? "💡 正在思考..." : "💡 思考完成"}</span>
      </button>

      {isOpen && (
        <div className="mt-2 whitespace-pre-wrap bg-white p-3 rounded-lg border border-gray-200">
          {displayText}
        </div>
      )}
    </div>
  );
};