import { ToolCallContentPartComponent } from '@assistant-ui/react';
import React, { useState } from 'react';
import { Button } from './button';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';

import { Wrench } from 'lucide-react';

const JsonCodeBlock = ({
  jsonString,
}: {
  jsonString: string | null | undefined;
}) => {
  return (
    // <pre className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-auto whitespace-pre-wrap break-words">
    //   <code ref={codeRef} className="language-json whitespace-pre-wrap break-words">
    //     {jsonString ?? ''}
    //   </code>
    // </pre>
    <pre className="overflow-x-auto bg-[#1e1e1e] text-[#d4d4d4] p-2 rounded-md font-mono text-sm leading-relaxed shadow-md border border-[#2d2d2d]">
      <code className="language-json whitespace-pre-wrap break-words">
        {jsonString ?? ''}
      </code>
    </pre>
  );
};
export const ToolFallback: ToolCallContentPartComponent = ({
  toolName,
  argsText,
  status,
  result,
}) => {
  if (status.type === 'running') {
    return (
      <div className="rounded mb-1 bg-muted/50 h-7 cursor-pointer hover:bg-muted/100 transition-colors">
        <Button
          variant="ghost"
          className="flex h-7 items-center gap-2 px-4 justify-start h-7 w-full text-gray-600 text-xs"
        >
          <Wrench className="size-4" /> 正在调用工具: {toolName}{' '}
          (请稍候，结果将会显示在这里)
        </Button>
      </div>
    );
  } else if (status.type === 'complete') {
    let parsedArgs: any = null;
    if (!argsText || argsText.trim() === '') {
      parsedArgs = argsText;
    } else if (argsText.trim() === '{}') {
      parsedArgs = {};
    } else {
      try {
        // 先尝试直接解析
        parsedArgs = JSON.parse(argsText);
      } catch (error) {
        // 如果直接解析失败，尝试修复格式后解析
        try {
          // 修复JSON格式
          const cleanedJson = argsText
            .replace(/(['"])?([a-zA-Z0-9_]+)(['"])?:/g, '"$2":') // 修复键未加引号
            .replace(/'/g, '"') // 替换单引号为双引号
            .replace(/(\w+):/g, '"$1":') // 修复未加引号的键
            .replace(/:\s*([^,"}\]]+)/g, (match, p1) =>
              isNaN(p1 as any) ? `:"${p1}"` : match,
            ); // 修复未加引号的字符串值

          parsedArgs = JSON.parse(cleanedJson);
        } catch (formatError) {
          console.error('JSON 解析失败:', formatError);
          parsedArgs = { error: '无效的JSON格式', raw: argsText };
        }
      }
    }

    const parsedResult = result?.content?.[0]?.text ?? result;
    // const parsedArgs = JSON.parse(argsText);

    return (
      <div className="rounded mb-1 bg-muted/50 h-7 cursor-pointer hover:bg-muted/100 transition-colors">
        <Sheet>
          <SheetTrigger asChild>
            <Button
              variant="ghost"
              className="flex items-center gap-2 justify-start h-7 w-full text-gray-600 text-xs"
            >
              {' '}
              <Wrench className="size-4" /> 完成工具调用: {toolName}{' '}
              (点击查看结果){' '}
            </Button>
          </SheetTrigger>
          <SheetContent side="right">
            <SheetHeader>
              <SheetTitle>工具调用结果</SheetTitle>
              <SheetDescription>工具名称: {toolName}</SheetDescription>
            </SheetHeader>
            <div className="flex flex-col gap-2 border-t pt-2 overflow-y-auto">
              <div className="px-4">
                <p className="font-semibold">工具调用参数:</p>
                <JsonCodeBlock
                  jsonString={
                    typeof parsedArgs === 'string'
                      ? parsedArgs
                      : JSON.stringify(parsedArgs, null, 2)
                  }
                />
              </div>

              {result !== undefined && (
                <div className="border-t border-dashed px-4 pt-2">
                  <p className="font-semibold">工具调用结果:</p>
                  <JsonCodeBlock
                    jsonString={
                      typeof parsedResult === 'string'
                        ? parsedResult
                        : JSON.stringify(parsedResult, null, 2)
                    }
                  />
                </div>
              )}
            </div>
          </SheetContent>
        </Sheet>
      </div>
    );
  }
};
