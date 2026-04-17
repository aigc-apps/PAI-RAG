'use client';

import { ToolCallContentPartComponent } from '@assistant-ui/react';
import React, { useState } from 'react';
import { ChevronRight, Wrench } from 'lucide-react';
import { Spinner } from '@/components/ui/loading';
import { useI18n } from '@/app/providers/i18n';

/** Parse loose JSON-ish strings emitted by LLMs. */
function safeParseJson(raw: string): unknown {
  if (!raw || !raw.trim()) return raw;
  if (raw.trim() === '{}') return {};
  try {
    return JSON.parse(raw);
  } catch {
    // Fallback: fix unquoted keys / single quotes, try again.
    try {
      const cleaned = raw
        .replace(/(['"])?([a-zA-Z0-9_]+)(['"])?:/g, '"$2":')
        .replace(/'/g, '"')
        .replace(/(\w+):/g, '"$1":')
        .replace(/:\s*([^,"}\]]+)/g, (m, v) =>
          isNaN(v as any) ? `:"${v}"` : m,
        );
      return JSON.parse(cleaned);
    } catch {
      return { error: 'Invalid JSON', raw };
    }
  }
}

function formatPayload(value: unknown): string {
  if (value === undefined || value === null) return '';
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

/** Compact monospace code block used inside the collapsed tool panel. */
const CodeBlock = ({ text }: { text: string }) => (
  <pre className="text-[11px] leading-relaxed font-mono text-slate-700 dark:text-slate-200 bg-slate-50 dark:bg-slate-900/60 rounded-md border border-slate-200 dark:border-slate-800 px-2.5 py-1.5 whitespace-pre-wrap break-words max-h-[280px] overflow-auto">
    {text}
  </pre>
);

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
      <div className="my-1 inline-flex items-center gap-1.5 rounded-md border border-primary/25 bg-primary/5 px-2 py-1 text-[11px] text-muted-foreground">
        <Spinner size="sm" />
        <Wrench className="w-3 h-3 text-muted-foreground" />
        <span>{t('chat.tools.callingTool')}</span>
        <span className="text-muted-foreground/60">·</span>
        <span className="font-mono text-foreground/90">{toolName}</span>
      </div>
    );
  }

  const parsedArgs = safeParseJson(argsText ?? '');
  const parsedResult =
    (result as any)?.content?.[0]?.text ?? result;

  return (
    <div className="my-1 rounded-md border border-primary/20 bg-primary/[0.04] overflow-hidden text-xs">
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
        <span className="font-mono text-[11px] text-foreground/90 truncate">
          {toolName}
        </span>
      </button>

      {open && (
        <div className="px-2 pb-2 pt-1 border-t border-primary/15 bg-background/60 space-y-2">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
              {t('chat.tools.toolArguments')}
            </p>
            <CodeBlock text={formatPayload(parsedArgs)} />
          </div>
          {result !== undefined && (
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                {t('chat.tools.toolResult')}
              </p>
              <CodeBlock text={formatPayload(parsedResult)} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};
