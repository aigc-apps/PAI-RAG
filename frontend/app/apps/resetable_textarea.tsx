"use client";

import * as React from "react";
import { useState } from "react";
import TextareaAutosize from "react-textarea-autosize";
import { RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";


// 自定义可重置的 Textarea 组件
interface ResettableTextareaProps
  extends React.ComponentProps<typeof TextareaAutosize> {
  value: string
  onChange: (event: React.ChangeEvent<HTMLTextAreaElement>) => void
  defaultValue: string
  placeholder?: string
  onReset: () => void
}

export const ResettableTextarea = React.forwardRef<
  HTMLTextAreaElement,
  ResettableTextareaProps
>(({ value, onChange, defaultValue, placeholder, onReset, className, ...props }, ref) => {
  return (
    <div className="relative">
      <TextareaAutosize
        ref={ref}
        value={value}
        onChange={(e) => {
          onChange(e);
        }}
        placeholder={placeholder}
        className={cn(
          "w-full min-h-[200px] max-h-96 overflow-y-auto rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          "resize-none", // 禁止手动拖拽
          className
        )}
        maxRows={20}
        {...props}
      />
      {/* 重置按钮 - 右下角 */}
      <Button
        type="button"
        variant="link"
        size="icon"
        className="absolute bottom-1 right-4 h-8 w-10 text-xs rounded-full opacity-70 hover:opacity-100 text-blue-700 hover:text-blue-800"
        onClick={onReset}
        title="重置为默认提示词"
      >
         重置
      </Button>
    </div>
  )
})