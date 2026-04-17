'use client';

import React from 'react';
import { usePathname, useRouter } from 'next/navigation';
import {
  Bot,
  Database,
  PlugZap,
  Search,
  Code,
  LassoSelectIcon,
  SquareActivity,
  GlobeLock,
  ShieldCheck,
  HardDrive,
  Settings,
} from 'lucide-react';
import { HeaderPortal } from '@/components/header-portal';
import { useI18n } from '@/app/providers/i18n';

interface ConfigNav {
  href: string;
  labelKey: string;
  icon: React.ReactNode;
}

const NAV_ITEMS: ConfigNav[] = [
  { href: '/config/model', labelKey: 'sidebar.model', icon: <Bot className="w-3.5 h-3.5" /> },
  { href: '/config/vectordb', labelKey: 'sidebar.vectordb', icon: <Database className="w-3.5 h-3.5" /> },
  { href: '/config/mcp', labelKey: 'sidebar.mcp', icon: <PlugZap className="w-3.5 h-3.5" /> },
  { href: '/config/search', labelKey: 'sidebar.search', icon: <Search className="w-3.5 h-3.5" /> },
  { href: '/config/code_sandbox', labelKey: 'sidebar.codeSandbox', icon: <Code className="w-3.5 h-3.5" /> },
  { href: '/config/chatdb', labelKey: 'sidebar.chatdb', icon: <LassoSelectIcon className="w-3.5 h-3.5" /> },
  { href: '/config/tracing', labelKey: 'sidebar.tracing', icon: <SquareActivity className="w-3.5 h-3.5" /> },
  { href: '/config/role', labelKey: 'sidebar.role', icon: <GlobeLock className="w-3.5 h-3.5" /> },
  { href: '/config/guardrail', labelKey: 'sidebar.guardrail', icon: <ShieldCheck className="w-3.5 h-3.5" /> },
  { href: '/config/cache', labelKey: 'sidebar.cache', icon: <HardDrive className="w-3.5 h-3.5" /> },
];

export default function ConfigLayout({ children }: { children: React.ReactNode }) {
  const { t } = useI18n();
  const pathname = usePathname();
  const router = useRouter();

  const activeItem =
    NAV_ITEMS.find((item) => pathname?.startsWith(item.href)) ?? NAV_ITEMS[0];
  const activeLabel = t(activeItem.labelKey);

  return (
    <div className="flex flex-col h-full min-h-0">
      <HeaderPortal>
        <div className="flex items-center gap-2">
          <Settings className="w-4 h-4 text-primary" />
          <h1 className="text-base font-semibold flex items-center gap-1.5">
            <span suppressHydrationWarning>{t('sidebar.settings')}</span>
            <span className="text-muted-foreground font-normal">·</span>
            <span suppressHydrationWarning>{activeLabel}</span>
          </h1>
        </div>
      </HeaderPortal>

      {/* Horizontal tab nav */}
      <div className="flex-none border-b border-border bg-background/60 backdrop-blur-sm">
        <div className="flex items-center gap-1 px-6 overflow-x-auto">
          {NAV_ITEMS.map((item) => {
            const isActive = pathname?.startsWith(item.href);
            return (
              <button
                key={item.href}
                type="button"
                onClick={() => router.push(item.href)}
                className={`flex items-center gap-1.5 text-xs py-2 px-3 border-b-2 -mb-px whitespace-nowrap transition-colors ${
                  isActive
                    ? 'border-primary text-foreground font-medium'
                    : 'border-transparent text-muted-foreground hover:text-foreground'
                }`}
              >
                {item.icon}
                <span suppressHydrationWarning>{t(item.labelKey)}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Content — uniform 24px padding for all config sub-pages */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="p-6 w-full">{children}</div>
      </div>
    </div>
  );
}
