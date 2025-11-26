'use client';

import React from 'react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { FilterIcon, Check } from 'lucide-react';
import { cn } from '@/lib/utils';

export type FileStatus = 'all' | 'succeeded' | 'failed' | 'pending' | 'parsing' | 'persisting';

interface FileStatusFilterProps {
  value: FileStatus;
  onValueChange: (value: FileStatus) => void;
  className?: string;
}

const statusOptions: Array<{ value: FileStatus; label: string; color: string }> = [
  { value: 'all', label: '全部', color: '' },
  { value: 'succeeded', label: '成功', color: 'text-green-500' },
  { value: 'failed', label: '失败', color: 'text-red-500' },
  { value: 'pending', label: '等待中', color: 'text-yellow-500' },
  { value: 'parsing', label: '解析中', color: 'text-blue-500' },
  { value: 'persisting', label: '索引中', color: 'text-blue-500' },
];

export function FileStatusFilter({ value, onValueChange, className }: FileStatusFilterProps) {
  const selectedOption = statusOptions.find(opt => opt.value === value) || statusOptions[0];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className={cn("h-6 w-[100px] bg-muted/50 hover:bg-muted justify-between text-xs", className)}
        >
          <span className={selectedOption.color}>{selectedOption.label}</span>
          <FilterIcon className="h-3 w-3 ml-1" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-[120px]">
        {statusOptions.map((option) => (
          <DropdownMenuItem
            key={option.value}
            onSelect={() => onValueChange(option.value)}
            className="text-xs"
          >
            <div className="flex items-center justify-between w-full">
              <span className={option.color}>{option.label}</span>
              {value === option.value && (
                <Check className="h-3 w-3 ml-2" />
              )}
            </div>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

