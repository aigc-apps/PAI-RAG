import { useEffect, useState, useRef } from "react";
import type { ReasoningContentPartComponent } from "@assistant-ui/react";

export const CollapsibleReasoning: ReasoningContentPartComponent = ({
  text,
  status,
}) => {
  const displayText = text || "";
  const isThinking = status.type === "running";
  const isComplete = status.type === "complete";

  const [isOpen, setIsOpen] = useState(isThinking);
  const hasAutoCollapsedRef = useRef(false); // 标记是否已经执行过自动收起

  useEffect(() => {
    if (isThinking) {
      setIsOpen(true);
      hasAutoCollapsedRef.current = false; 
    } else if (isComplete && !hasAutoCollapsedRef.current) {
      // 仅在第一次 complete 时触发延迟收起
      hasAutoCollapsedRef.current = true;
      const timer = setTimeout(() => {
        setIsOpen(false);
      }, 1000); // 延迟 1 秒收起

      return () => clearTimeout(timer);
    }
  }, [isThinking, isComplete]);

  if (!displayText && !isThinking) return null;

  const containerClassName = isOpen
    ? "bg-white p-3 rounded-lg border border-gray-200 text-xs text-gray-600 leading-relaxed"
    : "bg-white p-2 rounded-lg border border-gray-200 text-xs text-gray-600";

  return (
    <div className={containerClassName}>
      {isThinking && (
        <div className="whitespace-pre-wrap">
          <div className="text-gray-500 mb-2">💡 深度思考中</div>
          {displayText}
          {displayText === "" && <span className="text-gray-400">（生成中...）</span>}
        </div>
      )}

      {isComplete && (
        <>
          <button
            onClick={() => setIsOpen((prev) => !prev)}
            className="text-gray-600 hover:text-gray-900 flex items-center gap-1 w-full text-left"
          >
            💡 已完成思考 {isOpen ? "（点击收起）" : "（点击展开）"}
          </button>
          {isOpen && <div className="whitespace-pre-wrap mt-2">{displayText}</div>}
        </>
      )}
    </div>
  );
};