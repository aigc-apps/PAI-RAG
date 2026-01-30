'use client';
import { GlobeIcon } from '@radix-ui/react-icons';
import type { FC } from 'react';
import { makeAssistantToolUI } from '@assistant-ui/react';
import React, { useState, useEffect } from 'react';
import { PaperclipIcon, Search, FileSearch, FileText, ListTodoIcon, BookCheckIcon, Code2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Badge } from "@/components/ui/badge";
import { PhotoProvider, PhotoView } from "react-photo-view";
import ReactMarkdown from 'react-markdown';
import remarkGfm from "remark-gfm";
import { MarkdownRenderer } from '@/components/customized/markdown/markdown';
import { jsonrepair } from 'jsonrepair';
import { useI18n } from '@/app/providers/i18n';

// Helper function to safely parse JSON with error handling
const safeParseJSON = <T = any>(jsonString: string, fallback?: T): T => {
  try {
    return JSON.parse(jsonString) as T;
  } catch (error) {
    // If parsing fails, try to repair JSON using jsonrepair
    try {
      const repairedJson = jsonrepair(jsonString);
      return JSON.parse(repairedJson) as T;
    } catch (repairError) {
      console.error('Failed to parse JSON:', error, repairError, 'Original string:', jsonString);
      if (fallback !== undefined) {
        return fallback;
      }
      throw new Error('Invalid JSON format');
    }
  }
};

const JsonCodeBlock = ({
  jsonString,
}: {
  jsonString: string | null | undefined;
}) => {
  return (
    <pre className="overflow-x-auto bg-[#1e1e1e] text-[#d4d4d4] p-2 font-mono text-sm leading-relaxed shadow-md border border-[#2d2d2d]">
      <code className="language-json whitespace-pre-wrap break-words">
        {jsonString ?? ''}
      </code>
    </pre>
  );
};

const PythonCodeBlock = ({
  code,
}: {
  code: string | null | undefined;
}) => {
  return (
    <pre className="overflow-x-auto bg-[#1e1e1e] text-[#d4d4d4] p-3 rounded-md font-mono text-sm leading-relaxed shadow-md border border-[#2d2d2d]">
      <code className="language-python whitespace-pre-wrap break-words">
        {code ?? ''}
      </code>
    </pre>
  );
};

/* Search Web Tool UI */

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
    }[],
};


export const TavilySearchToolUI = makeAssistantToolUI<SearchWebArgs, string>({
  toolName: 'tavily-websearch',
  render: ({ args, status, result, isError }) => {
    const { t } = useI18n();
    console.log('TavilySearchTool 参数:', args);
    console.log('TavilySearchTool 状态:', status);

    if (status.type === 'running') {
      return (
        <div className="h-7 bg-muted/50 cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Button
            variant="ghost"
            className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
          >
            <GlobeIcon className="size-4" /> {t('chat.tools.searchingWeb')}: {args.query}{' '}
          </Button>
        </div>
      );
    } else if (status.type === 'complete') {
      if (!result || isError) {
        return (
          <div className="flex items-center gap-2 text-sm font-medium text-red-500">
            <GlobeIcon className="h-4 w-4" />
            <span>{t('chat.tools.searchFailed')}. {result || ''}</span>
          </div>
        );
      }
      let search_result: SearchWebResult;
      try {
        search_result = safeParseJSON<SearchWebResult>(result);
      } catch (error) {
        console.error('Failed to parse SearchWeb result:', error);
        search_result = { result: [] };
      }

      return (
        <div className="h-7 items-center bg-muted/50 cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Sheet>
            <SheetTrigger asChild>
              <Button
                variant="ghost"
                className="flex items-center justify-start h-7 w-full text-gray-600 text-xs gap-2 px-4 "
              >
                {' '}
                <GlobeIcon className="size-4" /> {t('chat.tools.webSearchComplete')}: {args.query}{' '}
              </Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetHeader>
                <SheetTitle>
                  {t('chat.tools.webSearchResults')} · {search_result?.result.length}
                </SheetTitle>
                <SheetDescription>{args.query}</SheetDescription>
              </SheetHeader>
              <div className="flex flex-col gap-2 border-t pt-2 pb-2 overflow-y-auto">
                <div className="pl-6 pr-2">
                  {search_result?.result.map((item, index) => (
                    <div
                      key={index}
                      className="text-sm p-3 hover:bg-muted/50 transition-colors"
                    >
                      <div className="flex flex-col gap-1 p-1 hover:bg-muted/50 transition-colors">
                        {/* Logo与标题行 */}
                        <div className="flex items-center gap-1">
                          <div className="flex-shrink-0 w-8 h-7 bg-muted flex items-center justify-center">
                            <img
                              src={item.favicon}
                              alt={item.hostname || ''}
                              className="w-5 h-5 object-cover rounded-sm"
                            />
                          </div>

                          {/* 标题链接 */}
                          <a
                            href={item.url}
                            className="font-medium text-foreground hover:text-primary hover:underline truncate transition-colors"
                          >
                            {item.title}
                          </a>
                        </div>

                        {/* 内容区域 */}
                        <p className="text-muted-foreground text-xs mt-1 leading-relaxed line-clamp-3">
                          {item.content}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      );
    }
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
    console.log('ChatDbTool 参数:', args);
    console.log('ChatDbTool 状态:', status);
    console.log('ChatDbTool 是否出错:', isError);
    console.log('ChatDbTool 结果:', result);

    if (status.type === 'running') {
      return (
        <div className="h-7 bg-muted/50 cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Button
            variant="ghost"
            className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
          >
            <GlobeIcon className="size-4" /> {t('chat.tools.queryingDb')}: {args.query}{' '}
          </Button>
        </div>
      );
    } else if (status.type === 'complete') {
      if (!result || isError) {
        return (
          <div className="flex items-center gap-2 text-sm font-medium text-red-500">
            <GlobeIcon className="h-4 w-4" />
            <span>{t('chat.tools.dbSearchFailed')}{result || ''}</span>
          </div>
        );
      }
      let db_result: ChatDbResult;
      try {
        db_result = safeParseJSON<ChatDbResult>(result);
      } catch (error) {
        console.error('Failed to parse ChatDb result:', error);
        db_result = { result: '', sql: '' };
      }

      return (
        <div className="h-7 items-center bg-muted/50 cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Sheet>
            <SheetTrigger asChild>
              <Button
                variant="ghost"
                className="flex items-center justify-start h-7 w-full text-gray-600 text-xs gap-2 px-4 "
              >
                {' '}
                <GlobeIcon className="size-4" /> {t('chat.tools.dbQueryComplete')}: {args.query}{' '}
              </Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetHeader>
                <SheetTitle>
                  {t('chat.tools.queryResults')}
                </SheetTitle>
                <SheetDescription>Query: {args.query}</SheetDescription>
              </SheetHeader>
              <div className="flex flex-col gap-2 border-t pt-2 pb-2 overflow-y-auto">
                <div className="pl-6 pr-2">
                <MarkdownRenderer content={`${t('chat.tools.dataResult')}

${db_result.result}

SQL:

\`\`\`sql
${db_result.sql}
\`\`\``} />

                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      );
    }
  },
});



export const PlanningToolUI = makeAssistantToolUI<SearchWebArgs, string>({
  toolName: 'planning-tool',
  render: ({ args, status, result, isError }) => {
    const { t } = useI18n();

    if (status.type === 'running') {
      return (
        <div className="h-7 bg-muted/50 cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Button
            variant="ghost"
            className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
          >
            <ListTodoIcon className="size-4" /> {t('chat.tools.makingPlan')}
          </Button>
        </div>
      );
    } else if (status.type === 'complete') {
      if (!result || isError) {
        return (
          <div className="flex items-center gap-2 text-sm font-medium text-red-500 mb-1">
            <ListTodoIcon className="h-4 w-4" />
            <span>{t('chat.tools.planFailed')} {result || ''}</span>
          </div>
        );
      }
      
      let plan_result;
      try {
        plan_result = safeParseJSON(result);
      } catch (error) {
        console.error('Failed to parse plan result JSON:', error);
        return (
          <div className="flex items-center gap-2 text-sm font-medium text-red-500 mb-1">
            <ListTodoIcon className="h-4 w-4" />
            <span>{t('chat.tools.parsePlanFailed')}</span>
          </div>
        );
      }
      return (
        <div className="h-7 bg-muted/50 cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Sheet>
            <SheetTrigger asChild>
              <Button
                variant="ghost"
                className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
              >
                {' '}
                <ListTodoIcon className="size-4" /> {t('chat.tools.planComplete')}
              </Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetHeader>
                <SheetTitle>
                  {t('chat.tools.executionPlan')} - {t('common.total')}{plan_result?.steps.length}{t('chat.tools.steps')} 
                </SheetTitle>
                <SheetDescription>{args.query}</SheetDescription>
              </SheetHeader>
              <div className="flex flex-col gap-2 border-t pt-2 pb-2 overflow-y-auto">
                <div className="pl-3 pr-2">
                  {plan_result?.steps.map((item: any, index: number) => (
                    <div
                      key={index}
                      className="text-sm p-1 hover:bg-muted/50 transition-colors py-3"
                    >
                        {/* 标题 */}
                        <div
                          className="font-medium text-foreground hover:text-primary transition-colors bg-gray-50 p-1 rounded"
                        >
                          {index + 1}. {item}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      );
    }
  },
});

export const SearchWebToolUI = makeAssistantToolUI<SearchWebArgs, string>({
  toolName: 'aliyun-websearch',
  render: ({ args, status, result, isError }) => {
    const { t } = useI18n();
    console.log('SearchWebToolUI 参数:', args);
    console.log('SearchWebToolUI 状态:', status);

    if (status.type === 'running') {
      return (
        <div className="h-7 bg-muted/50 cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Button
            variant="ghost"
            className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
          >
            <GlobeIcon className="size-4" /> {t('chat.tools.searchingWeb')}: {args.query}{' '}
          </Button>
        </div>
      );
    } else if (status.type === 'complete') {
      if (!result || isError) {
        return (
          <div className="flex items-center gap-2 text-sm font-medium text-red-400">
            <GlobeIcon className="h-4 w-4" />
            <span>{t('chat.tools.searchFailed')} {result || ''}</span>
          </div>
        );
      }
      let search_result: SearchWebResult;
      try {
        search_result = safeParseJSON<SearchWebResult>(result);
      } catch (error) {
        console.error('Failed to parse SearchWeb result:', error);
        search_result = { result: [] };
      }
      return (
        <div className="h-7 bg-muted/50 cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Sheet>
            <SheetTrigger asChild>
              <Button
                variant="ghost"
                className="h-7 flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
              >
                {' '}
                <GlobeIcon className="size-4" /> {t('chat.tools.webSearchComplete')}: {args.query}{' '}
              </Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetHeader>
                <SheetTitle>
                  {t('chat.tools.webSearchResults')} · {search_result?.result.length}
                </SheetTitle>
                <SheetDescription>{args.query}</SheetDescription>
              </SheetHeader>
              <div className="flex flex-col gap-2 border-t pt-2 pb-2 overflow-y-auto">
                <div className="pl-6 pr-2">
                  {search_result?.result.map((item, index) => (
                    <div
                      key={index}
                      className="text-sm p-3 hover:bg-muted/50 transition-colors"
                    >
                      <div className="flex flex-col gap-1 p-1 hover:bg-muted/50 transition-colors">
                        {/* Logo与标题行 */}
                        <div className="flex items-center gap-1">
                          <div className="flex-shrink-0 w-8 h-7 bg-muted flex items-center justify-center">
                            <img
                              src={item.favicon}
                              alt={item.hostname || ''}
                              className="w-5 h-5 object-cover rounded-sm"
                            />
                          </div>

                          {/* 标题链接 */}
                          <a
                            href={item.url}
                            className="font-medium text-foreground hover:text-primary hover:underline truncate transition-colors"
                          >
                            {item.title}
                          </a>
                        </div>

                        {/* 内容区域 */}
                        <p className="text-muted-foreground text-xs mt-1 leading-relaxed line-clamp-3">
                          {item.content}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      );
    }
  },
});


/* Read File Tool UI */

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
        <div className="h-7 bg-muted/50  cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Button
            variant="ghost"
            className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
          >
            <PaperclipIcon className="size-4" /> {t('chat.tools.readingFile')}: {args.file_id}
          </Button>
        </div>
      );
    } else if (status.type === 'complete') {
      if (!result || isError) {
        return null;
      }

      let parsedResult;
      try {
        parsedResult = safeParseJSON(result);
      } catch (error) {
        console.error('Failed to parse ReadFile result:', error);
        return null;
      }
      console.log('ReadFileToolUI 结果:', parsedResult);
      console.log('ReadFileToolUI 参数:', args);

      return (
        <div className="h-7 bg-muted/50  cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Sheet>
            <SheetTrigger asChild>
              <Button
                variant="ghost"
                className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
              >
                <PaperclipIcon className="size-4" /> {t('chat.tools.fileReadComplete')}: {args.file_name}
              </Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetHeader>
                <SheetTitle>{t('chat.tools.fileReadResults')}</SheetTitle>
                <SheetDescription>{t('chat.tools.fileName')}:{args.file_name}</SheetDescription>
              </SheetHeader>
              <div className="flex flex-col gap-2 border-t pt-2 pb-2 overflow-y-auto">
                <div className="border-t border-dashed px-4 pt-2">
                  <p className="font-semibold">{t('chat.tools.fileReadResult')}:</p>
                  <JsonCodeBlock
                    jsonString={
                      typeof parsedResult === 'string'
                        ? parsedResult
                        : JSON.stringify(parsedResult, null, 2)
                    }
                  />
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      );
    }
  },
});

/* Search File Tool UI */

export type SearchFileToolArgs = {
  query_str: string;
};

export const SearchFileToollUI = makeAssistantToolUI<
  SearchFileToolArgs,
  string
>({
  toolName: 'search-file',
  render: ({ args, status, result, isError }) => {
    const { t } = useI18n();
    console.log('SearchFileToollUI 参数:', args);

    if (status.type === 'running') {
      return (
        <div className="h-7 bg-muted/50  cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Button
            variant="ghost"
            className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
          >
            <FileSearch className="size-4" /> {t('chat.tools.searchingFile')}: {args.query_str}
          </Button>
        </div>
      );
    } else if (status.type === 'complete') {
      if (!result || isError) {
        return null;
      }
      let parsedResult;
      try {
        parsedResult = safeParseJSON(result);
      } catch (error) {
        console.error('Failed to parse SearchFile result:', error);
        return null;
      }
      console.log('SearchFileToollUI 结果:', parsedResult);

      return (
        <div className="h-7 bg-muted/50  cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Sheet>
            <SheetTrigger asChild>
              <Button
                variant="ghost"
                className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
              >
                <FileSearch className="size-4" /> {t('chat.tools.fileSearchComplete')}: {args.query_str}
              </Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetHeader>
                <SheetTitle>{t('chat.tools.fileSearchResults')}</SheetTitle>
                <SheetDescription>{t('chat.tools.searchQuery')}:{args.query_str}</SheetDescription>
              </SheetHeader>
              <div className="flex flex-col gap-2 border-t pt-2 pb-2 overflow-y-auto">
                <div className="border-t border-dashed px-4 pt-2">
                  <p className="font-semibold">{t('chat.tools.fileSearchResult')}:</p>
                  <JsonCodeBlock
                    jsonString={
                      typeof parsedResult === 'string'
                        ? parsedResult
                        : JSON.stringify(parsedResult, null, 2)
                    }
                  />
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      );
    }
  },
});

/* Search Knowledgebase Tool UI */

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
    images: {
      url: string;
      desc: string;
    }[];

  }[];
  error: string;
};

export const SearchKbToolUI = makeAssistantToolUI<SearchKbArgs, string>({
  toolName: "search-knowledgebase",
  render: ({ args, status, result, isError }) => {
    const { t } = useI18n();
    console.log("SearchKbToolUI 参数:", args);
    console.log("SearchKbToolUI 状态:", status);

    if (status.type === "running") {
      return (
        <div className="h-7 bg-muted/50 cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Button
            variant="ghost"
            className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
          >
            <BookCheckIcon className="size-4" /> {t('chat.tools.searchingKb')}: {args.query}{" "}
          </Button>
        </div>
      );
    } else if (status.type === "complete") {
      if (!result || isError) {
        return (
          <div className="flex items-center gap-2 text-sm font-medium text-red-500">
            <GlobeIcon className="h-4 w-4" />
            <span>{t('chat.tools.searchFailed')} {result || ''}</span>
          </div>
        );
      }
      let search_result: SearchKbResult;
      try {
        search_result = safeParseJSON<SearchKbResult>(result);
      } catch (error) {
        console.error('Failed to parse SearchKb result:', error);
        search_result = { result: [], error: '' };
      }
      return (
        <div className="h-7 bg-muted/50 cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors round-sm">
          <Sheet>
            <SheetTrigger asChild>
              <Button
                variant="ghost"
                className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
              >
                {" "}
                <BookCheckIcon className="size-4" /> {t('chat.tools.kbSearchComplete')}: {args.query}{" "}
              </Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetHeader>
                <SheetTitle>{t('chat.tools.kbSearchResults')}</SheetTitle>
                <SheetDescription>{args.query}</SheetDescription>
              </SheetHeader>
              <div className="flex flex-col gap-2 border-t pt-2 pb-2 overflow-y-auto max-h-[calc(100vh-120px)]">
                <div className="pl-4 pr-2">
                  <Accordion
                    type="single"
                    collapsible
                    className="max-w-lg my-4 w-full space-y-2"
                  >
                    {search_result?.result.map((item, index) => (
                      <AccordionItem
                        key={index}
                        value={`item-${index}`}
                        className="border px-4"
                      >
                        <AccordionTrigger>
                          Chunk{index + 1}: {item.title}{" "} 
                          <Badge className="bg-green-600/10 dark:bg-green-600/20 hover:bg-green-600/10 text-green-500 shadow-none rounded-full">
                            {parseFloat(item.score).toFixed(4)}
                          </Badge>
                        </AccordionTrigger>
                        <AccordionContent>
                          <a href={item.url} className='text-blue-600 hover:underline'>{t('chat.tools.documentLink')}</a>
                          <div>{item.content}</div>
                          
                          {item?.images.map((meta, index) => (
                            <PhotoProvider
                              key={index}
                              maskOpacity={0.8}
                              overlayRender={({}) => {
                                return (
                                  <div className="absolute left-0 bottom-0 p-4 w-full min-h-30 text-sm text-slate-300 z-50 bg-black/50">
                                    <div>{t('chat.tools.imageDesc')}:{meta.desc}</div>
                                  </div>
                                );
                              }}
                            >
                              <PhotoView key={index} src={meta.url}>
                                <img src={meta.url} className="w-10 h-10" />
                              </PhotoView>
                            </PhotoProvider>
                          ))}
                        </AccordionContent>
                      </AccordionItem>
                    ))}
                  </Accordion>
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      );
    }
  },
});

/* Python Interpreter Tool UI */

export type PythonInterpreterArgs = {
  code: string;
};

export const PythonInterpreterToolUI = makeAssistantToolUI<PythonInterpreterArgs, string>({
  toolName: 'PythonInterpreter',
  render: ({ args, status, result, isError }) => {
    const { t } = useI18n();
    const [isOpen, setIsOpen] = useState(false);
    const [savedCode, setSavedCode] = useState('');
    const [codeGenerated, setCodeGenerated] = useState(false); // Mark whether code generation is complete

    // When status changes from running to complete, first expand to show results, then auto-collapse after 3 seconds
    useEffect(() => {
      if (status.type === 'complete') {
        setIsOpen(true); // Expand when complete to show results
        // Auto-collapse after 3 seconds
        const timer = setTimeout(() => {
          setIsOpen(false);
        }, 3000);
        return () => clearTimeout(timer);
      } else if (status.type === 'running') {
        // Expand immediately when running, don't wait for code
        setIsOpen(true);
        // Reset code generation status
        setCodeGenerated(false);
      }
    }, [status.type]);

    // Extract code and save the last valid code value (using useMemo for performance optimization)
    const extractCode = React.useMemo(() => {
      if (!args) return '';
      
      // Check if it's an empty object
      if (typeof args === 'object' && Object.keys(args).length === 0) {
        return '';
      }
      
      if (typeof args === 'object' && 'code' in args) {
        const code = typeof args.code === 'string' ? args.code : String(args.code || '');
        // Only return when code is not empty
        if (code && code.trim()) {
          return code;
        }
      } else if (typeof args === 'string') {
        const argsString: string = args as string; // Store as string to preserve type
        try {
          const parsed = safeParseJSON(argsString);
          if (parsed && typeof parsed === 'object' && 'code' in parsed) {
            const code = typeof parsed.code === 'string' ? parsed.code : String(parsed.code || '');
            if (code && code.trim()) {
              return code;
            }
          }
          // If not JSON format, might be direct code string
          if (argsString.trim() && argsString !== '{}') {
            return argsString;
          }
        } catch {
          // If parsing fails, might be direct code string
          if (argsString.trim() && argsString !== '{}') {
            return argsString;
          }
        }
      }
      return '';
    }, [args]);

    // Update saved code (only update when there's valid code)
    useEffect(() => {
      if (extractCode && extractCode.trim()) {
        setSavedCode(extractCode);
      }
    }, [extractCode]);

    // Use saved code or current code (prefer current code, use saved code if empty)
    const codeToShow = extractCode && extractCode.trim() ? extractCode : savedCode;

    // Detect if code generation is complete (code is stable and no longer changing)
    useEffect(() => {
      if (status.type === 'running' && codeToShow && codeToShow.trim()) {
        // After 1.5 seconds delay, consider code generation complete (code hasn't changed during this time)
        // Timer resets with each code change, only considered complete after code is stable for 1.5 seconds
        const timer = setTimeout(() => {
          setCodeGenerated(true);
        }, 1500);
        return () => clearTimeout(timer);
      } else {
        // If code is empty or status is not running, reset status
        setCodeGenerated(false);
      }
    }, [codeToShow, status.type]);

    if (status.type === 'running') {
      return (
        <div className="mb-1 rounded transition-colors">
          <Collapsible open={isOpen} onOpenChange={setIsOpen}>
            <div className="h-7 bg-muted/50 cursor-pointer hover:bg-muted/100 rounded transition-colors">
              <CollapsibleTrigger asChild>
                <Button
                  variant="ghost"
                  className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
                >
                  <Code2 className="size-4" /> {t('chat.tools.callingTool')}: PythonInterpreter
                </Button>
              </CollapsibleTrigger>
            </div>
            <CollapsibleContent className="mt-2">
              <div className="rounded-md border border-border bg-background p-2 space-y-3">
                <div>
                  <p className="text-xs font-semibold mb-2 text-muted-foreground">{t('chat.tools.generatingCode')}:</p>
                  {codeToShow && codeToShow.trim() ? (
                    <PythonCodeBlock code={codeToShow} />
                  ) : (
                    <div className="bg-muted/30 rounded p-3 text-sm text-muted-foreground italic">
                      {t('chat.tools.codeLoading')}
                    </div>
                  )}
                </div>
                {/* Show running prompt if code generation is complete */}
                {codeToShow && codeToShow.trim() && codeGenerated && (
                  <div className="bg-muted/50 rounded p-3 text-xs font-semibold text-muted-foreground flex items-center gap-2">
                    <span>⚙️</span>
                    <span>{t('chat.tools.codeRunning')}</span>
                  </div>
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>
        </div>
      );
    } else if (status.type === 'complete') {
      const parsedResult = (typeof result === 'object' && result !== null && 'content' in result) 
        ? (result as any).content?.[0]?.text ?? result 
        : result;
      
      return (
        <div className="mb-1 rounded transition-colors">
          <Collapsible open={isOpen} onOpenChange={setIsOpen}>
            <div className="h-7 bg-muted/50 cursor-pointer hover:bg-muted/100 rounded transition-colors">
              <CollapsibleTrigger asChild>
                <Button
                  variant="ghost"
                  className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
                >
                  <Code2 className="size-4" /> {t('chat.tools.toolCallComplete')}: PythonInterpreter
                </Button>
              </CollapsibleTrigger>
            </div>
            <CollapsibleContent className="mt-2">
              <div className="rounded-md border border-border bg-background p-2 space-y-3">
                <div>
                  <p className="text-xs font-semibold mb-2 text-muted-foreground">{t('chat.tools.generatingCode')}:</p>
                  {codeToShow && codeToShow.trim() ? (
                    <PythonCodeBlock code={codeToShow} />
                  ) : (
                    <div className="bg-muted/30 rounded p-3 text-sm text-muted-foreground italic">
                      {t('chat.tools.codeNotProvided')}
                    </div>
                  )}
                </div>
                {parsedResult !== undefined && (
                  <div className="border-t border-dashed pt-3">
                    <p className="text-xs font-semibold mb-2 text-muted-foreground">{t('chat.tools.executionResult')}:</p>
                    <div className="bg-muted/30 rounded p-2 text-sm whitespace-pre-wrap break-words">
                      {typeof parsedResult === 'string' ? parsedResult : JSON.stringify(parsedResult, null, 2)}
                    </div>
                  </div>
                )}
                {isError && (
                  <div className="border-t border-dashed pt-3">
                    <p className="text-xs font-semibold mb-2 text-red-500">{t('chat.tools.error')}:</p>
                    <div className="bg-red-50 dark:bg-red-950/20 rounded p-2 text-sm text-red-600 dark:text-red-400 whitespace-pre-wrap break-words">
                      {typeof parsedResult === 'string' ? parsedResult : JSON.stringify(parsedResult, null, 2)}
                    </div>
                  </div>
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>
        </div>
      );
    }
    return null;
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
