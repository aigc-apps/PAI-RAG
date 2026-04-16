'use client';

import React from 'react';
import { Globe } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';
import { useI18n } from '@/app/providers/i18n';

interface LanguageSwitcherProps {
  variant?: 'icon' | 'inline';
}

export function LanguageSwitcher({ variant = 'inline' }: LanguageSwitcherProps) {
  const { language, setLanguage } = useI18n();
  const label = language === 'zh' ? '中文' : 'EN';

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className={
            variant === 'icon'
              ? 'h-7 w-7 p-0 text-muted-foreground hover:text-foreground'
              : 'h-7 px-2 text-xs gap-1.5 text-muted-foreground hover:text-foreground'
          }
          aria-label="Switch language"
        >
          <Globe className="h-3.5 w-3.5" />
          {variant !== 'icon' && <span>{label}</span>}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent side="top" align="end" className="menu-compact">
        <DropdownMenuItem
          onClick={() => setLanguage('zh')}
          className={language === 'zh' ? 'bg-accent' : ''}
        >
          中文
        </DropdownMenuItem>
        <DropdownMenuItem
          onClick={() => setLanguage('en')}
          className={language === 'en' ? 'bg-accent' : ''}
        >
          English
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
