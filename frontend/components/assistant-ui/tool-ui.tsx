'use client';
import { GlobeIcon } from '@radix-ui/react-icons';
import type { FC, ReactNode } from 'react';
import { makeAssistantToolUI } from '@assistant-ui/react';
import React, { useState, useEffect } from 'react';
import {
  PaperclipIcon,
  FileSearch,
  ListTodoIcon,
  BookCheckIcon,
  Code2,
  ChevronRight,
  AlertCircle,
} from 'lucide-react';
import { Spinner } from '@/components/ui/loading';
import { Badge } from '@/components/ui/badge';
import { PhotoProvider, PhotoView } from 'react-photo-view';
import { MarkdownRenderer } from '@/components/customized/markdown/markdown';
import { ToolContent } from '@/components/ui/tool-content';
import { jsonrepair } from 'jsonrepair';
import { useI18n } from '@/app/providers/i18n';

// ===== Helpers =====

const safeParseJSON = <T = any,>(jsonString: string, fallback?: T): T => {
  try {
    return JSON.parse(jsonString) as T;
  } catch {
    try {
      return JSON.parse(jsonrepair(jsonString)) as T;
    } catch (repairError) {
      console.error('Failed to parse JSON:', repairError, 'Original:', jsonString);
      if (fallback !== undefined) return fallback;
      throw new Error('Invalid JSON format');
    }
  }
};

/** Compact monospace code block, uniform across all tool UIs. */
const CodeBlock = ({
  text,
  language,
}: {
  text: string;
  language?: string;
}) => (
  <div className="rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/60 overflow-hidden">
    {language && (
      <div className="px-2.5 py-1 border-b border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-slate-900 text-[10px] uppercase tracking-wider text-muted-foreground font-mono">
        {language}
      </div>
    )}
    <pre className="text-[11px] leading-relaxed font-mono text-slate-700 dark:text-slate-200 px-2.5 py-1.5 whitespace-pre-wrap break-words max-h-[320px] overflow-auto">
      {text}
    </pre>
  </div>
);

// ===== Shared bubble primitives =====

// Shared visual treatment so every tool card (running / collapsible / error)
// reads as "the agent's aside" — a small indented panel with a left accent
// that's clearly not body text but also doesn't steal focus from the reply.
const BUBBLE_BASE =
  'border-l-2 border-l-primary/60 border-y border-r border-primary/20 bg-primary/[0.04]';

/** The tight running bubble: spinner + tool label + query. */
const RunningBubble: FC<{
  icon: ReactNode;
  label: string;
  detail?: string;
}> = ({ icon, label, detail }) => (
  <div className={`my-1 inline-flex items-center gap-1.5 rounded-md ${BUBBLE_BASE} px-2 py-1 text-[11px] text-muted-foreground`}>
    <Spinner size="sm" />
    {icon}
    <span>{label}</span>
    {detail !== undefined && detail !== '' && (
      <>
        <span className="text-muted-foreground/60">·</span>
        <span className="font-mono text-muted-foreground truncate max-w-[220px]">
          {detail}
        </span>
      </>
    )}
  </div>
);

/** The inline collapsible bubble: header row + expand-in-place content. */
const CollapsibleBubble: FC<{
  icon: ReactNode;
  label: string;
  detail?: string;
  defaultOpen?: boolean;
  children: ReactNode;
}> = ({ icon, label, detail, defaultOpen = false, children }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={`my-1 rounded-md ${BUBBLE_BASE} overflow-hidden text-xs`}>
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
        {icon}
        <span className="text-[11px] text-muted-foreground shrink-0">{label}</span>
        {detail !== undefined && detail !== '' && (
          <span className="font-mono text-[11px] text-muted-foreground truncate">
            · {detail}
          </span>
        )}
      </button>
      {open && (
        <div className="px-2 pb-2 pt-1 border-t border-primary/15 bg-background/60 space-y-2">
          {children}
        </div>
      )}
    </div>
  );
};

/** Inline error bubble. */
const ErrorBubble: FC<{ icon: ReactNode; label: string; detail?: string }> = ({
  icon,
  label,
  detail,
}) => (
  <div className="my-1 inline-flex items-center gap-1.5 rounded-md border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-[11px] text-rose-600 dark:text-rose-400">
    <AlertCircle className="w-3 h-3" />
    {icon}
    <span>{label}</span>
    {detail && <span className="truncate max-w-[260px]">· {detail}</span>}
  </div>
);

// Small icon-coloring helper
const DimIcon = ({ children }: { children: ReactNode }) => (
  <span className="text-muted-foreground shrink-0 [&_svg]:w-3 [&_svg]:h-3">
    {children}
  </span>
);

// ===== Tool UIs =====

export type SearchWebArgs = {
  query: string;
};

type SearchWebResult = {
  result: {
    title: string;
    content: string;
    url: string;
    favicon: string;
    hostname: string;
    publish_time: string;
    score: string;
  }[];
};

const WebSearchResults: FC<{ items: SearchWebResult['result'] }> = ({ items }) => {
  // Dedup by URL + title — some backends echo the same result multiple times
  // (e.g. aggregating different providers), which should show as one card.
  const deduped = React.useMemo(() => {
    if (!items?.length) return [];
    const seen = new Set<string>();
    const out: SearchWebResult['result'] = [];
    for (const item of items) {
      const key = `${item.url || ''}|${item.title || ''}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(item);
    }
    return out;
  }, [items]);

  if (!deduped.length) {
    return <p className="text-[11px] text-muted-foreground">—</p>;
  }
  return (
    <div className="space-y-1.5">
      {deduped.map((item, index) => (
        <a
          key={index}
          href={item.url}
          target="_blank"
          rel="noreferrer"
          className="flex items-start gap-2 rounded-md px-1.5 py-1 hover:bg-muted/60 transition-colors"
        >
          <div className="w-5 h-5 rounded bg-muted flex items-center justify-center shrink-0 mt-0.5 overflow-hidden">
            {item.favicon ? (
              <img
                src={item.favicon}
                alt={item.hostname || ''}
                className="w-3.5 h-3.5 object-cover"
              />
            ) : (
              <GlobeIcon className="w-3 h-3 text-muted-foreground" />
            )}
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-xs font-medium text-foreground hover:text-primary truncate">
              {item.title}
            </div>
            {item.content && (
              <p className="text-[11px] text-muted-foreground line-clamp-2 mt-0.5">
                {item.content}
              </p>
            )}
          </div>
        </a>
      ))}
    </div>
  );
};

export const TavilySearchToolUI = makeAssistantToolUI<SearchWebArgs, string>({
  toolName: 'tavily-websearch',
  render: ({ args, status, result, isError }) => {
    const { t } = useI18n();

    if (status.type === 'running') {
      return (
        <RunningBubble
          icon={<DimIcon><GlobeIcon /></DimIcon>}
          label={t('chat.tools.searchingWeb')}
          detail={args.query}
        />
      );
    }
    if (status.type === 'complete') {
      if (!result || isError) {
        return (
          <ErrorBubble
            icon={<DimIcon><GlobeIcon /></DimIcon>}
            label={t('chat.tools.searchFailed')}
            detail={String(result || '')}
          />
        );
      }
      let data: SearchWebResult;
      try {
        data = safeParseJSON<SearchWebResult>(result);
      } catch {
        data = { result: [] };
      }
      return (
        <CollapsibleBubble
          icon={<DimIcon><GlobeIcon /></DimIcon>}
          label={t('chat.tools.webSearchComplete')}
          detail={args.query}
        >
          <WebSearchResults items={data.result} />
        </CollapsibleBubble>
      );
    }
    return null;
  },
});

export type ChatDbArgs = {
  query: string;
};

type ChatDbResult = {
  result: string;
  sql: string;
};

export const ChatDbToolUI = makeAssistantToolUI<ChatDbArgs, string>({
  toolName: 'chat-db',
  render: ({ args, status, result, isError }) => {
    const { t } = useI18n();

    if (status.type === 'running') {
      return (
        <RunningBubble
          icon={<DimIcon><GlobeIcon /></DimIcon>}
          label={t('chat.tools.queryingDb')}
          detail={args.query}
        />
      );
    }
    if (status.type === 'complete') {
      if (!result || isError) {
        return (
          <ErrorBubble
            icon={<DimIcon><GlobeIcon /></DimIcon>}
            label={t('chat.tools.dbSearchFailed')}
            detail={String(result || '')}
          />
        );
      }
      let data: ChatDbResult;
      try {
        data = safeParseJSON<ChatDbResult>(result);
      } catch {
        data = { result: '', sql: '' };
      }
      return (
        <CollapsibleBubble
          icon={<DimIcon><GlobeIcon /></DimIcon>}
          label={t('chat.tools.dbQueryComplete')}
          detail={args.query}
        >
          <div className="prose prose-sm max-w-none text-xs">
            <MarkdownRenderer
              content={`${t('chat.tools.dataResult')}\n\n${data.result}\n\nSQL:\n\n\`\`\`sql\n${data.sql}\n\`\`\``}
            />
          </div>
        </CollapsibleBubble>
      );
    }
    return null;
  },
});

export const PlanningToolUI = makeAssistantToolUI<SearchWebArgs, string>({
  toolName: 'planning-tool',
  render: ({ args, status, result, isError }) => {
    const { t } = useI18n();

    if (status.type === 'running') {
      return (
        <RunningBubble
          icon={<DimIcon><ListTodoIcon /></DimIcon>}
          label={t('chat.tools.makingPlan')}
        />
      );
    }
    if (status.type === 'complete') {
      if (!result || isError) {
        return (
          <ErrorBubble
            icon={<DimIcon><ListTodoIcon /></DimIcon>}
            label={t('chat.tools.planFailed')}
            detail={String(result || '')}
          />
        );
      }
      let plan: any;
      try {
        plan = safeParseJSON(result);
      } catch {
        return (
          <ErrorBubble
            icon={<DimIcon><ListTodoIcon /></DimIcon>}
            label={t('chat.tools.parsePlanFailed')}
          />
        );
      }
      const steps: string[] = plan?.steps ?? [];
      return (
        <CollapsibleBubble
          icon={<DimIcon><ListTodoIcon /></DimIcon>}
          label={t('chat.tools.planComplete')}
          detail={`${steps.length} ${t('chat.tools.steps')}`}
        >
          <ol className="space-y-1">
            {steps.map((item, index) => (
              <li
                key={index}
                className="flex items-start gap-2 text-xs rounded-md px-1.5 py-1 hover:bg-muted/40"
              >
                <span className="text-[10px] font-mono text-muted-foreground shrink-0 mt-0.5">
                  {String(index + 1).padStart(2, '0')}
                </span>
                <span className="flex-1">{item}</span>
              </li>
            ))}
          </ol>
        </CollapsibleBubble>
      );
    }
    return null;
  },
});

export const SearchWebToolUI = makeAssistantToolUI<SearchWebArgs, string>({
  toolName: 'aliyun-websearch',
  render: ({ args, status, result, isError }) => {
    const { t } = useI18n();

    if (status.type === 'running') {
      return (
        <RunningBubble
          icon={<DimIcon><GlobeIcon /></DimIcon>}
          label={t('chat.tools.searchingWeb')}
          detail={args.query}
        />
      );
    }
    if (status.type === 'complete') {
      if (!result || isError) {
        return (
          <ErrorBubble
            icon={<DimIcon><GlobeIcon /></DimIcon>}
            label={t('chat.tools.searchFailed')}
            detail={String(result || '')}
          />
        );
      }
      let data: SearchWebResult;
      try {
        data = safeParseJSON<SearchWebResult>(result);
      } catch {
        data = { result: [] };
      }
      return (
        <CollapsibleBubble
          icon={<DimIcon><GlobeIcon /></DimIcon>}
          label={t('chat.tools.webSearchComplete')}
          detail={args.query}
        >
          <WebSearchResults items={data.result} />
        </CollapsibleBubble>
      );
    }
    return null;
  },
});

// ===== Read File =====

export type ReadFileToolArgs = {
  file_id: string;
  file_name: string;
};

export const ReadFileToollUI = makeAssistantToolUI<ReadFileToolArgs, string>({
  toolName: 'read-file',
  render: ({ args, status, result, isError }) => {
    const { t } = useI18n();

    if (status.type === 'running') {
      return (
        <RunningBubble
          icon={<DimIcon><PaperclipIcon /></DimIcon>}
          label={t('chat.tools.readingFile')}
          detail={args.file_id}
        />
      );
    }
    if (status.type === 'complete') {
      if (!result || isError) return null;
      return (
        <CollapsibleBubble
          icon={<DimIcon><PaperclipIcon /></DimIcon>}
          label={t('chat.tools.fileReadComplete')}
          detail={args.file_name}
        >
          <ToolContent value={result} />
        </CollapsibleBubble>
      );
    }
    return null;
  },
});

// ===== Search File =====

export type SearchFileToolArgs = {
  query_str: string;
};

export const SearchFileToollUI = makeAssistantToolUI<SearchFileToolArgs, string>({
  toolName: 'search-file',
  render: ({ args, status, result, isError }) => {
    const { t } = useI18n();

    if (status.type === 'running') {
      return (
        <RunningBubble
          icon={<DimIcon><FileSearch /></DimIcon>}
          label={t('chat.tools.searchingFile')}
          detail={args.query_str}
        />
      );
    }
    if (status.type === 'complete') {
      if (!result || isError) return null;
      return (
        <CollapsibleBubble
          icon={<DimIcon><FileSearch /></DimIcon>}
          label={t('chat.tools.fileSearchComplete')}
          detail={args.query_str}
        >
          <ToolContent value={result} />
        </CollapsibleBubble>
      );
    }
    return null;
  },
});

// ===== Search Knowledge Base =====

export type SearchKbArgs = {
  query: string;
};

type SearchKbResult = {
  result: {
    title: string;
    content: string;
    url: string;
    favicon: string;
    hostname: string;
    publish_time: string;
    score: string;
    images: { url: string; desc: string }[];
  }[];
  error: string;
};

export const SearchKbToolUI = makeAssistantToolUI<SearchKbArgs, string>({
  toolName: 'search-knowledgebase',
  render: ({ args, status, result, isError }) => {
    const { t } = useI18n();

    if (status.type === 'running') {
      return (
        <RunningBubble
          icon={<DimIcon><BookCheckIcon /></DimIcon>}
          label={t('chat.tools.searchingKb')}
          detail={args.query}
        />
      );
    }
    if (status.type === 'complete') {
      if (!result || isError) {
        return (
          <ErrorBubble
            icon={<DimIcon><BookCheckIcon /></DimIcon>}
            label={t('chat.tools.searchFailed')}
            detail={String(result || '')}
          />
        );
      }
      let data: SearchKbResult;
      try {
        data = safeParseJSON<SearchKbResult>(result);
      } catch {
        data = { result: [], error: '' };
      }
      const chunks = data?.result ?? [];
      return (
        <CollapsibleBubble
          icon={<DimIcon><BookCheckIcon /></DimIcon>}
          label={t('chat.tools.kbSearchComplete')}
          detail={args.query}
        >
          {chunks.length === 0 ? (
            <p className="text-[11px] text-muted-foreground">—</p>
          ) : (
            <div className="space-y-1.5">
              {chunks.map((item, index) => (
                <KbChunkItem key={index} item={item} index={index} />
              ))}
            </div>
          )}
        </CollapsibleBubble>
      );
    }
    return null;
  },
});

const KbChunkItem: FC<{
  item: SearchKbResult['result'][number];
  index: number;
}> = ({ item, index }) => {
  const [open, setOpen] = useState(false);
  const { t } = useI18n();
  return (
    <div className="rounded-md border border-border bg-background">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-2 py-1.5 text-left hover:bg-muted/40"
      >
        <ChevronRight
          className={`w-3 h-3 text-muted-foreground shrink-0 transition-transform ${
            open ? 'rotate-90' : ''
          }`}
        />
        <Badge
          variant="secondary"
          className="h-4 px-1 text-[9px] font-mono shrink-0"
        >
          #{index + 1}
        </Badge>
        <span className="text-xs font-medium truncate flex-1">{item.title}</span>
        <Badge className="bg-green-500/10 hover:bg-green-500/10 text-green-700 border-green-500/30 shadow-none h-4 px-1 text-[10px] font-mono shrink-0">
          {parseFloat(item.score).toFixed(3)}
        </Badge>
      </button>
      {open && (
        <div className="px-3 pb-2 pt-1 border-t border-border/50 space-y-1.5 text-[11px] leading-relaxed">
          {item.url && (
            <a
              href={item.url}
              target="_blank"
              rel="noreferrer"
              className="text-primary hover:underline inline-block"
            >
              {t('chat.tools.documentLink')}
            </a>
          )}
          <div className="whitespace-pre-wrap break-words text-foreground/80">
            {item.content}
          </div>
          {item.images?.length > 0 && (
            <div className="flex gap-1 flex-wrap pt-1">
              {item.images.map((meta, idx) => (
                <PhotoProvider
                  key={idx}
                  maskOpacity={0.8}
                  overlayRender={() => (
                    <div className="absolute left-0 bottom-0 p-3 w-full min-h-20 text-xs text-slate-300 z-50 bg-black/50">
                      {t('chat.tools.imageDesc')}: {meta.desc}
                    </div>
                  )}
                >
                  <PhotoView src={meta.url}>
                    <img
                      src={meta.url}
                      className="w-8 h-8 rounded object-cover cursor-pointer"
                    />
                  </PhotoView>
                </PhotoProvider>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ===== Python Interpreter =====
// Keeps special streaming behaviour: auto-expand on run, auto-collapse 3s
// after complete (users can still toggle manually).

export type PythonInterpreterArgs = {
  code: string;
};

export const PythonInterpreterToolUI = makeAssistantToolUI<PythonInterpreterArgs, string>({
  toolName: 'PythonInterpreter',
  render: ({ args, status, result, isError }) => {
    const { t } = useI18n();
    const [open, setOpen] = useState(false);
    const [savedCode, setSavedCode] = useState('');

    useEffect(() => {
      if (status.type === 'running') {
        setOpen(true);
      } else if (status.type === 'complete') {
        setOpen(true);
        const timer = setTimeout(() => setOpen(false), 3000);
        return () => clearTimeout(timer);
      }
    }, [status.type]);

    // Extract code progressively from streaming args
    const extractCode = React.useMemo(() => {
      if (!args) return '';
      if (typeof args === 'object' && 'code' in args) {
        const code = typeof args.code === 'string' ? args.code : String(args.code || '');
        return code && code.trim() ? code : '';
      }
      if (typeof args === 'string') {
        const s = args as string;
        try {
          const parsed = safeParseJSON(s);
          if (parsed && typeof parsed === 'object' && 'code' in parsed) {
            const code = typeof parsed.code === 'string' ? parsed.code : String(parsed.code || '');
            if (code && code.trim()) return code;
          }
        } catch {
          // fall through
        }
        if (s.trim() && s !== '{}') return s;
      }
      return '';
    }, [args]);

    useEffect(() => {
      if (extractCode && extractCode.trim()) setSavedCode(extractCode);
    }, [extractCode]);

    const codeToShow = extractCode && extractCode.trim() ? extractCode : savedCode;
    const parsedResult =
      typeof result === 'object' && result !== null && 'content' in result
        ? (result as any).content?.[0]?.text ?? result
        : result;

    const isRunning = status.type === 'running';
    const label = isRunning
      ? t('chat.tools.callingTool')
      : t('chat.tools.toolCallComplete');

    return (
      <div className="my-1 rounded-md border border-border bg-muted/30 overflow-hidden text-xs">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="w-full flex items-center gap-1.5 px-2 py-1.5 hover:bg-muted/60 transition-colors text-left"
        >
          {isRunning ? (
            <Spinner size="sm" />
          ) : (
            <ChevronRight
              className={`w-3 h-3 text-muted-foreground shrink-0 transition-transform ${
                open ? 'rotate-90' : ''
              }`}
            />
          )}
          <DimIcon><Code2 /></DimIcon>
          <span className="text-[11px] text-muted-foreground shrink-0">{label}</span>
          <span className="font-mono text-[11px] text-foreground/90 truncate">
            · PythonInterpreter
          </span>
        </button>

        {open && (
          <div className="px-2 pb-2 pt-1 border-t border-border/60 space-y-2">
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                {isRunning ? t('chat.tools.generatingCode') : t('chat.tools.generatingCode')}
              </p>
              {codeToShow && codeToShow.trim() ? (
                <CodeBlock text={codeToShow} language="python" />
              ) : (
                <div className="text-[11px] text-muted-foreground italic bg-muted/40 rounded px-2.5 py-1.5 border border-border">
                  {isRunning ? t('chat.tools.codeLoading') : t('chat.tools.codeNotProvided')}
                </div>
              )}
            </div>
            {!isRunning && parsedResult !== undefined && !isError && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                  {t('chat.tools.executionResult')}
                </p>
                <CodeBlock
                  text={
                    typeof parsedResult === 'string'
                      ? parsedResult
                      : JSON.stringify(parsedResult, null, 2)
                  }
                />
              </div>
            )}
            {!isRunning && isError && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-destructive mb-1">
                  {t('chat.tools.error')}
                </p>
                <pre className="text-[11px] leading-relaxed font-mono bg-destructive/10 text-destructive rounded-md border border-destructive/30 px-2.5 py-1.5 whitespace-pre-wrap break-words max-h-[280px] overflow-auto">
                  {typeof parsedResult === 'string'
                    ? parsedResult
                    : JSON.stringify(parsedResult, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    );
  },
});

const ToolUIWrapper: FC = () => {
  return (
    <>
      <PlanningToolUI />
      <TavilySearchToolUI />
      <SearchWebToolUI />
      <ReadFileToollUI />
      <SearchFileToollUI />
      <SearchKbToolUI />
      <ChatDbToolUI />
      <PythonInterpreterToolUI />
    </>
  );
};

export default ToolUIWrapper;
