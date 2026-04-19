'use client';

import { ToolCallContentPartComponent } from '@assistant-ui/react';
import React, { useState } from 'react';
import { ArrowDownToDot, ArrowUpFromDot, ChevronRight, Wrench } from 'lucide-react';
import { Spinner } from '@/components/ui/loading';
import { ToolContent } from '@/components/ui/tool-content';
import { useI18n } from '@/app/providers/i18n';

export const ToolFallback: ToolCallContentPartComponent = ({
  toolName,
  argsText,
  status,
  result,
}) => {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);

  const isRunning = status.type === 'running';

  if (isRunning) {
    return (
      <div className="my-1 inline-flex items-center gap-1.5 rounded-md border-l-2 border-l-primary/60 border-y border-r border-primary/20 bg-primary/[0.04] pl-2 pr-2 py-1 text-[11px] text-muted-foreground">
        <Spinner size="sm" />
        <Wrench className="w-3 h-3 text-muted-foreground" />
        <span>{t('chat.tools.callingTool')}</span>
        <span className="text-muted-foreground/60">·</span>
        <span className="font-mono text-muted-foreground">{toolName}</span>
      </div>
    );
  }

  // Tool results from our agent are usually wrapped like
  // `{content: [{text: "<actual payload>"}]}` (LlamaIndex ToolOutput shape).
  // Peel that off here so the smart renderer sees the real payload.
  const parsedResult =
    (result as any)?.content?.[0]?.text ?? result;

  return (
    <div className="my-1 rounded-md border-l-2 border-l-primary/60 border-y border-r border-primary/20 bg-primary/[0.04] overflow-hidden text-xs">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-1.5 px-2 py-1.5 hover:bg-primary/10 transition-colors text-left"
      >
        <ChevronRight
          className={`w-3 h-3 text-primary/70 shrink-0 transition-transform ${
            open ? 'rotate-90' : ''
          }`}
        />
        <Wrench className="w-3 h-3 text-muted-foreground shrink-0" />
        <span className="text-[11px] text-muted-foreground shrink-0">
          {t('chat.tools.toolCallComplete')}:
        </span>
        <span className="font-mono text-[11px] text-muted-foreground truncate">
          {toolName}
        </span>
      </button>

      {open && (
        <div className="px-2 pb-2 pt-1 border-t border-primary/15 bg-background/60 space-y-2">
          <div>
            <p className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
              <ArrowDownToDot className="w-3 h-3" />
              {t('chat.tools.toolArguments')}
            </p>
            <ToolContent value={argsText ?? ''} />
          </div>
          {result !== undefined && (
            <div>
              <p className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-emerald-600 dark:text-emerald-400 mb-1">
                <ArrowUpFromDot className="w-3 h-3" />
                {t('chat.tools.toolResult')}
              </p>
              <ToolContent value={parsedResult} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};
