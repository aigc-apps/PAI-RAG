"use client";
import { Calendar, Home, Inbox, Search, Settings, Bot, Wrench } from "lucide-react";
import React from "react";
import { useState } from "react";
import Link from "next/link";
import {
    Sidebar,
    SidebarContent,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
    SidebarHeader,
} from "@/components/ui/sidebar";
import {
    Avatar,
    AvatarImage,
  } from "@/components/ui/avatar"
import { useRouter } from 'next/navigation';
import { usePathname } from 'next/navigation';

// Menu items.
const items = [
    {
        title: "Home",
        url: "#",
        icon: Home,
    },
    {
        title: "Inbox",
        url: "#",
        icon: Inbox,
    },
    {
        title: "Calendar",
        url: "#",
        icon: Calendar,
    },
    {
        title: "Search",
        url: "#",
        icon: Search,
    },
    {
        title: "Settings",
        url: "#",
        icon: Settings,
    },
]

export function AppSidebar() {
    const router = useRouter();
    const pathname = usePathname();
    return (
        <Sidebar side='left'>
            <SidebarHeader>
                <div className="flex items-center space-x-2">
                    <Avatar>
                        <AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" />
                    </Avatar>
                    <span className="text-lg font-medium">Agent Workspace</span>
                </div>
                
            </SidebarHeader>
            <SidebarContent>
                <SidebarGroup>
                    <SidebarGroupLabel>系统设置</SidebarGroupLabel>
                    <SidebarGroupContent className="gap-4">
                        <SidebarMenu className="gap-4">
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild onClick={() => router.push('/')} isActive={pathname === '/'} >
                                    <Link href="/">
                                        <Home />
                                        <span>对话</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild onClick={() => router.push('/config/llm')} isActive={pathname === '/config/llm'}>
                                    <Link href="/config/llm">
                                        <Bot />
                                        <span>LLM配置</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild onClick={() => router.push('/config/mcp')} isActive={pathname === '/config/mcp'}>
                                    <Link href="/config/mcp">
                                        <Wrench />
                                        <span>MCP配置</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild onClick={() => router.push('/config/search')} isActive={pathname === '/config/search'}>
                                    <Link href="/config/search">
                                        <Search />
                                        <span>搜索配置</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>
            </SidebarContent>
        </Sidebar>
    )
}
