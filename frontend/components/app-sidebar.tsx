'use client';
import { Search, Settings, Bot, Wrench, Database, PlugZap, SquareActivity, GlobeLock, ShieldCheck, Code } from 'lucide-react';
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
import { Avatar, AvatarImage } from '@/components/ui/avatar';
import { ThreadList } from '@/components/assistant-ui/thread-list';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuItem,
} from '@/components/ui/dropdown-menu';
import {
  ChevronUp,
  MessageCircle,
  ChevronDown,
  BookIcon,
  AppWindowIcon,
  Scale
} from 'lucide-react';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import Link from 'next/link';

export function AppSidebar() {
  return (
    <Sidebar side="left">
      <SidebarHeader>
        <div className="flex items-center space-x-2">
          <Avatar>
            <AvatarImage
              src="https://pai-rag.oss-cn-hangzhou.aliyuncs.com/logo/pairag_1.png"
              alt="@shadcn"
            />
          </Avatar>
          <span className="text-lg font-medium">PAI-RAG</span>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarMenu>
          <Collapsible defaultOpen className="group/collapsible">
            <SidebarMenuItem>
              <SidebarMenuButton asChild>
                <Link href="/knowledgebases"><BookIcon /> 知识库</Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <SidebarMenuButton asChild>
                <Link href="/apps"><AppWindowIcon /> 应用</Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <SidebarMenuButton asChild>
                <Link href="/evaluation"><Scale /> 评估</Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <CollapsibleTrigger asChild>
                <SidebarMenuButton>
                    <MessageCircle />
                    <span>对话</span>
                    <ChevronDown className="ml-auto" />
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
        <SidebarMenu className="gap-4">
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton>
                  <Settings /> Settings
                  <ChevronUp className="ml-auto" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent side="top" className="w-50">
                <DropdownMenuItem asChild>
                  <Link href="/config/model"><Bot /> 模型</Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/vectordb"><Database /> 向量数据库 </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/mcp"><PlugZap /> MCP</Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/search"><Search /> 搜索</Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/code_sandbox"><Code /> Code沙箱</Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/tracing"><SquareActivity /> 链路追踪</Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/role"><GlobeLock /> 权限控制 </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/guardrail"><ShieldCheck /> 安全护栏 </Link>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  );
}
