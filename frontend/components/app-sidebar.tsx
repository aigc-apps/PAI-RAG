'use client';
import {
  ChevronDown,
  MessageCircle,
  BookIcon,
  AppWindowIcon,
  Scale,
  Settings,
} from 'lucide-react';
import React from 'react';
import {
  Sidebar,
  SidebarContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarRail,
  SidebarFooter,
  SidebarMenuSub,
} from '@/components/ui/sidebar';
import { ThreadList } from '@/components/assistant-ui/thread-list';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import Link from 'next/link';
import { useI18n } from '@/app/providers/i18n';
import { LanguageSwitcher } from '@/components/language-switcher';

export function AppSidebar() {
  const { t } = useI18n();

  return (
    <Sidebar side="left">
      <SidebarHeader>
        <div className="flex items-center gap-2 py-1">
          <div className="logo-icon flex items-center justify-center w-6 h-6 rounded-md font-semibold text-[11px] shrink-0">
            P
          </div>
          <span className="text-[13px] font-semibold tracking-tight flex-1 truncate">
            PAI-RAG
          </span>
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarMenu>
          <Collapsible defaultOpen className="group/collapsible">
            <SidebarMenuItem>
              <SidebarMenuButton asChild size="sm">
                <Link href="/knowledgebases">
                  <BookIcon />
                  <span suppressHydrationWarning>{t('sidebar.knowledgebase')}</span>
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <SidebarMenuButton asChild size="sm">
                <Link href="/apps">
                  <AppWindowIcon />
                  <span suppressHydrationWarning>{t('sidebar.apps')}</span>
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <SidebarMenuButton asChild size="sm">
                <Link href="/evaluation">
                  <Scale />
                  <span suppressHydrationWarning>{t('sidebar.evaluation')}</span>
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <CollapsibleTrigger asChild>
                <SidebarMenuButton size="sm">
                  <MessageCircle />
                  <span suppressHydrationWarning>{t('sidebar.conversation')}</span>
                  <ChevronDown className="ml-auto h-3.5 w-3.5 transition-transform group-data-[state=open]/collapsible:rotate-180" />
                </SidebarMenuButton>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <SidebarMenuSub>
                  <ThreadList />
                </SidebarMenuSub>
              </CollapsibleContent>
            </SidebarMenuItem>
          </Collapsible>
        </SidebarMenu>
      </SidebarContent>

      <SidebarRail />

      <SidebarFooter>
        <div className="flex items-center gap-1">
          <SidebarMenuButton asChild size="sm" className="flex-1">
            <Link href="/config/model">
              <Settings />
              <span suppressHydrationWarning>{t('sidebar.settings')}</span>
            </Link>
          </SidebarMenuButton>
          <LanguageSwitcher variant="icon" />
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
