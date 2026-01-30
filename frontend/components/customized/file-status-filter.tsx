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
import { useI18n } from '@/app/providers/i18n';


export type FileStatus = 'all' | 'succeeded' | 'failed' | 'pending' | 'parsing' | 'persisting';

interface FileStatusFilterProps {
  value: FileStatus;
  onValueChange: (value: FileStatus) => void;
  className?: string;
}

export function FileStatusFilter({ value, onValueChange, className }: FileStatusFilterProps) {
  const { t } = useI18n();
  const statusOptions: Array<{ value: FileStatus; label: string; color: string }> = [
    { value: 'all', label: t('knowledgebase.statusAll'), color: '' },
    { value: 'succeeded', label: t('knowledgebase.parseSuccess'), color: 'text-green-500' },
    { value: 'failed', label: t('knowledgebase.parseFailed'), color: 'text-red-500' },
    { value: 'pending', label: t('knowledgebase.pendingParse'), color: 'text-yellow-500' },
    { value: 'parsing', label: t('knowledgebase.parsing'), color: 'text-blue-500' },
    { value: 'persisting', label: t('knowledgebase.persisting'), color: 'text-blue-500' },
  ];
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

