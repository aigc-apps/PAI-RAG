'use client';
import { GlobeIcon } from '@radix-ui/react-icons';
import type { FC } from 'react';
import { makeAssistantToolUI } from '@assistant-ui/react';
import React, { useState, useEffect } from 'react';
import { PaperclipIcon, Search, FileSearch, FileText, ListTodoIcon, BookCheckIcon } from 'lucide-react';
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
import { Badge } from "@/components/ui/badge";
import { PhotoProvider, PhotoView } from "react-photo-view";
import ReactMarkdown from 'react-markdown';
import remarkGfm from "remark-gfm";
import { MarkdownRenderer } from '@/components/customized/markdown/markdown';
import { jsonrepair } from 'jsonrepair';

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
    console.log('TavilySearchTool 参数:', args);
    console.log('TavilySearchTool 状态:', status);

    if (status.type === 'running') {
      return (
        <div className="h-7 bg-muted/50 cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Button
            variant="ghost"
            className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
          >
            <GlobeIcon className="size-4" /> 正在搜索网页中: {args.query}{' '}
          </Button>
        </div>
      );
    } else if (status.type === 'complete') {
      if (!result || isError) {
        return (
          <div className="flex items-center gap-2 text-sm font-medium text-red-500">
            <GlobeIcon className="h-4 w-4" />
            <span>未能获取搜索结果. {result || ''}</span>
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
                <GlobeIcon className="size-4" /> 完成网页搜索: {args.query}{' '}
              </Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetHeader>
                <SheetTitle>
                  网页搜索结果 · {search_result?.result.length}
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
            <GlobeIcon className="size-4" /> 正在查询数据库: {args.query}{' '}
          </Button>
        </div>
      );
    } else if (status.type === 'complete') {
      if (!result || isError) {
        return (
          <div className="flex items-center gap-2 text-sm font-medium text-red-500">
            <GlobeIcon className="h-4 w-4" />
            <span>未能获取数据库结果。{result || ''}</span>
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
                <GlobeIcon className="size-4" /> 完成数据库查询: {args.query}{' '}
              </Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetHeader>
                <SheetTitle>
                  查询结果
                </SheetTitle>
                <SheetDescription>Query: {args.query}</SheetDescription>
              </SheetHeader>
              <div className="flex flex-col gap-2 border-t pt-2 pb-2 overflow-y-auto">
                <div className="pl-6 pr-2">
                <MarkdownRenderer content={`数据结果\n\n${db_result.result}\n\nSQL:\n\n\`\`\`sql\n${db_result.sql}\n\`\`\``} />

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

    if (status.type === 'running') {
      return (
        <div className="h-7 bg-muted/50 cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Button
            variant="ghost"
            className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
          >
            <ListTodoIcon className="size-4" /> 正在制定执行计划
          </Button>
        </div>
      );
    } else if (status.type === 'complete') {
      if (!result || isError) {
        return (
          <div className="flex items-center gap-2 text-sm font-medium text-red-500 mb-1">
            <ListTodoIcon className="h-4 w-4" />
            <span>制定计划失败 {result || ''}</span>
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
            <span>解析计划结果失败</span>
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
                <ListTodoIcon className="size-4" /> 执行计划完成
              </Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetHeader>
                <SheetTitle>
                  执行计划 - 共{plan_result?.steps.length}步 
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
    console.log('SearchWebToolUI 参数:', args);
    console.log('SearchWebToolUI 状态:', status);

    if (status.type === 'running') {
      return (
        <div className="h-7 bg-muted/50 cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Button
            variant="ghost"
            className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
          >
            <GlobeIcon className="size-4" /> 正在搜索网页中: {args.query}{' '}
          </Button>
        </div>
      );
    } else if (status.type === 'complete') {
      if (!result || isError) {
        return (
          <div className="flex items-center gap-2 text-sm font-medium text-red-400">
            <GlobeIcon className="h-4 w-4" />
            <span>未能获取搜索结果 {result || ''}</span>
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
                <GlobeIcon className="size-4" /> 完成网页搜索: {args.query}{' '}
              </Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetHeader>
                <SheetTitle>
                  网页搜索结果 · {search_result?.result.length}
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
    if (status.type === 'running') {
      return (
        <div className="h-7 bg-muted/50  cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Button
            variant="ghost"
            className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
          >
            <PaperclipIcon className="size-4" /> 正在进行文件读取: {args.file_name}
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
                <PaperclipIcon className="size-4" /> 完成文件读取: {args.file_name}
              </Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetHeader>
                <SheetTitle>文件读取结果</SheetTitle>
                <SheetDescription>文件名：{args.file_name}</SheetDescription>
              </SheetHeader>
              <div className="flex flex-col gap-2 border-t pt-2 pb-2 overflow-y-auto">
                <div className="border-t border-dashed px-4 pt-2">
                  <p className="font-semibold">文件读取结果:</p>
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
    console.log('SearchFileToollUI 参数:', args);

    if (status.type === 'running') {
      return (
        <div className="h-7 bg-muted/50  cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Button
            variant="ghost"
            className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
          >
            <FileSearch className="size-4" /> 正在进行文件搜索: {args.query_str}
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
                <FileSearch className="size-4" /> 完成文件搜索: {args.query_str}
              </Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetHeader>
                <SheetTitle>文件搜索结果</SheetTitle>
                <SheetDescription>搜索问题：{args.query_str}</SheetDescription>
              </SheetHeader>
              <div className="flex flex-col gap-2 border-t pt-2 pb-2 overflow-y-auto">
                <div className="border-t border-dashed px-4 pt-2">
                  <p className="font-semibold">文件搜索结果:</p>
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
    console.log("SearchKbToolUI 参数:", args);
    console.log("SearchKbToolUI 状态:", status);

    if (status.type === "running") {
      return (
        <div className="h-7 bg-muted/50 cursor-pointer mb-1 hover:bg-muted/100 rounded transition-colors">
          <Button
            variant="ghost"
            className="flex items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
          >
            <BookCheckIcon className="size-4" /> 正在搜索知识库中: {args.query}{" "}
          </Button>
        </div>
      );
    } else if (status.type === "complete") {
      if (!result || isError) {
        return (
          <div className="flex items-center gap-2 text-sm font-medium text-red-500">
            <GlobeIcon className="h-4 w-4" />
            <span>未能获取搜索结果 {result || ''}</span>
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
                <BookCheckIcon className="size-4" /> 完成知识库搜索: {args.query}{" "}
              </Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetHeader>
                <SheetTitle>知识库搜索结果</SheetTitle>
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
                          <a href={item.url} className='text-blue-600 hover:underline'>document link</a>
                          <div>{item.content}</div>
                          
                          {item?.images.map((meta, index) => (
                            <PhotoProvider
                              key={index}
                              maskOpacity={0.8}
                              overlayRender={({}) => {
                                return (
                                  <div className="absolute left-0 bottom-0 p-4 w-full min-h-30 text-sm text-slate-300 z-50 bg-black/50">
                                    <div>图片描述：{meta.desc}</div>
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
    </>
  );
};

export default ToolUIWrapper;
