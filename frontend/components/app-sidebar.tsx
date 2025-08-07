"use client";
import { Search, Settings, Bot, Wrench, UserCog } from "lucide-react";
import React from "react";
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
} from "@/components/ui/sidebar";
import { Avatar, AvatarImage } from "@/components/ui/avatar";
import { ThreadList } from "@/components/assistant-ui/thread-list";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import {
  ChevronUp,
  MessageCircle,
  ChevronDown,
  BookIcon,
  AppWindowIcon,
} from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
export function AppSidebar({
  activeTab,
  setActiveTab,
}: {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}) {
  return (
    <Sidebar side="left">
      <SidebarHeader>
        <div className="flex items-center space-x-2">
          <Avatar>
            <AvatarImage
              src="https://pai-rag.oss-cn-hangzhou.aliyuncs.com/logo/pairag.png"
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
              <SidebarMenuButton onClick={() => setActiveTab("/knowledgebase")}>
                <BookIcon />
                <span>知识库</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <SidebarMenuButton onClick={() => setActiveTab("/chatbot")}>
                <AppWindowIcon />
                <span>应用</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <CollapsibleTrigger asChild>
                <SidebarMenuButton onClick={() => setActiveTab("/")}>
                  <MessageCircle />
                  <span>对话</span>
                  <ChevronDown className="ml-auto" />
                </SidebarMenuButton>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <SidebarMenuSub onClick={() => setActiveTab("/")}>
                  <ThreadList />
                </SidebarMenuSub>
                {/* <ThreadList /> */}
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
                <DropdownMenuItem onClick={() => setActiveTab("/config/model")}>
                  <Bot /> <span>Model</span>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setActiveTab("/config/mcp")}>
                  <Wrench /> <span>MCP</span>
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => setActiveTab("/config/search")}
                >
                  <Search /> <span>搜索</span>
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => setActiveTab("/config/tracing")}
                >
                  <Wrench /> <span>链路追踪</span>
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => setActiveTab("/config/prompts")}
                >
                  <UserCog /> <span>Prompt</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  );
}
