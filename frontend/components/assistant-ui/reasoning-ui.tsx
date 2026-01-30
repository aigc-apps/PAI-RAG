import { useEffect, useState, useRef } from "react";
import type { ReasoningContentPartComponent } from "@assistant-ui/react";
import { useI18n } from '@/app/providers/i18n';

export const CollapsibleReasoning: ReasoningContentPartComponent = ({
  text,
  status,
}) => {
  const { t } = useI18n();
  const displayText = text || "";
  const isThinking = status.type === "running";
  const isComplete = status.type === "complete";

  const [isOpen, setIsOpen] = useState(isThinking);
  const hasAutoCollapsedRef = useRef(false); // Mark whether auto-collapse has been executed

  useEffect(() => {
    if (isThinking) {
      setIsOpen(true);
      hasAutoCollapsedRef.current = false; 
    } else if (isComplete && !hasAutoCollapsedRef.current) {
      // Only trigger delayed collapse on first complete
      hasAutoCollapsedRef.current = true;
      const timer = setTimeout(() => {
        setIsOpen(false);
      }, 1000); // Delay 1 second before collapsing

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
          <div className="text-gray-500 mb-2">💡 {t('chat.reasoning.deepThinking')}</div>
          {displayText}
          {displayText === "" && <span className="text-gray-400">{t('chat.reasoning.generating')}</span>}
        </div>
      )}

      {isComplete && (
        <>
          <button
            onClick={() => setIsOpen((prev) => !prev)}
            className="text-gray-600 hover:text-gray-900 flex items-center gap-1 w-full text-left"
          >
            💡 {t('chat.reasoning.thinkingComplete')} {isOpen ? t('chat.reasoning.clickToCollapse') : t('chat.reasoning.clickToExpand')}
          </button>
          {isOpen && <div className="whitespace-pre-wrap mt-2">{displayText}</div>}
        </>
      )}
    </div>
  );
};