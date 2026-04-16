'use client';

import * as React from 'react';
import TextareaAutosize from 'react-textarea-autosize';
import { RotateCcw, FileCode2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface ResettableTextareaProps
  extends React.ComponentProps<typeof TextareaAutosize> {
  value: string;
  onChange: (event: React.ChangeEvent<HTMLTextAreaElement>) => void;
  defaultValue: string;
  placeholder?: string;
  onReset: () => void;
  resetLabel?: string;
  language?: string;
}

export const ResettableTextarea = React.forwardRef<
  HTMLTextAreaElement,
  ResettableTextareaProps
>(
  (
    {
      value,
      onChange,
      defaultValue,
      placeholder,
      onReset,
      resetLabel = '重置默认',
      language = 'Markdown',
      className,
      ...props
    },
    ref,
  ) => {
    const charCount = value?.length ?? 0;
    const isModified = value !== defaultValue;

    return (
      <div
        className={cn(
          'flex flex-col rounded-lg border border-input bg-background overflow-hidden',
          'focus-within:border-primary/50 focus-within:ring-2 focus-within:ring-primary/15',
          'transition-colors',
        )}
      >
        {/* Toolbar */}
        <div className="flex items-center justify-between px-3 py-1.5 border-b border-border bg-muted/40 text-xs">
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <FileCode2 className="w-3.5 h-3.5" />
            <span className="font-medium">{language}</span>
            {isModified && (
              <span className="text-[10px] text-amber-600 dark:text-amber-400">
                · 已修改
              </span>
            )}
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-xs text-muted-foreground hover:text-primary"
            onClick={onReset}
            disabled={!isModified}
          >
            <RotateCcw className="w-3 h-3 mr-1" />
            {resetLabel}
          </Button>
        </div>

        {/* Editor */}
        <TextareaAutosize
          ref={ref}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          className={cn(
            'w-full px-4 py-3 text-sm leading-relaxed',
            'font-mono text-foreground placeholder:text-muted-foreground',
            'bg-transparent border-0 outline-none focus:outline-none',
            'resize-none',
            className,
          )}
          minRows={12}
          maxRows={24}
          {...props}
        />

        {/* Footer with char count */}
        <div className="flex items-center justify-end px-3 py-1 border-t border-border bg-muted/30 text-[11px] text-muted-foreground">
          {charCount.toLocaleString()} 字符
        </div>
      </div>
    );
  },
);
ResettableTextarea.displayName = 'ResettableTextarea';
