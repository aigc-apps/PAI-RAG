'use client';
import { Search, Settings, Bot, Wrench, Database, PlugZap, SquareActivity, GlobeLock, ShieldCheck, Code, Users, Plus, Check, X } from 'lucide-react';
import React, { useState } from 'react';
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
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from '@/components/ui/dropdown-menu';
import {
  LassoSelectIcon,
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
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import Link from 'next/link';
import { useTenant } from '@/app/providers/tenant';

export function AppSidebar() {
  const { tenantId, tenantName, tenants, setTenant, addTenant, removeTenant } = useTenant();
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [newTenantId, setNewTenantId] = useState('');
  const [newTenantName, setNewTenantName] = useState('');

  const handleCreateTenant = () => {
    if (newTenantId.trim() && newTenantName.trim()) {
      addTenant(newTenantId.trim(), newTenantName.trim());
      setNewTenantId('');
      setNewTenantName('');
      setIsCreateDialogOpen(false);
      // 创建并切换到新工作空间后跳转到首页
      setTimeout(() => window.location.href = '/', 100);
    }
  };

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
        {/* 工作空间选择器 */}
        <div className="mt-1">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="w-full justify-between text-xs h-8">
                <div className="flex items-center gap-2 truncate">
                  <Users className="h-3 w-3 flex-shrink-0" />
                  <span className="truncate">{tenantName}</span>
                </div>
                <ChevronDown className="h-3 w-3 flex-shrink-0" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56">
              <DropdownMenuLabel className="text-xs text-muted-foreground">选择工作空间</DropdownMenuLabel>
              <DropdownMenuSeparator />
              {tenants.map((tenant) => (
                <DropdownMenuItem
                  key={tenant.id}
                  onClick={() => {
                    if (tenant.id !== tenantId) {
                      setTenant(tenant.id, tenant.name);
                      // 切换工作空间后跳转到首页
                      setTimeout(() => window.location.href = '/', 100);
                    }
                  }}
                >
                  <div className="flex items-center gap-2">
                    {tenant.id === tenantId && <Check className="h-3 w-3 text-primary" />}
                    {tenant.id !== tenantId && <div className="w-3" />}
                    <span className="text-xs">{tenant.name}</span>
                  </div>
                  <span className="text-xs text-muted-foreground">{tenant.id}</span>
                </DropdownMenuItem>
              ))}
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="cursor-pointer"
                onClick={() => setIsCreateDialogOpen(true)}
              >
                <Plus className="h-3 w-3 mr-2" />
                <span className="text-xs">新建工作空间</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {/* 创建工作空间对话框 */}
        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>新建工作空间</DialogTitle>
              <DialogDescription>创建一个新的工作空间来隔离数据</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="tenant-id">工作空间 ID</Label>
                <Input
                  id="tenant-id"
                  placeholder="请输入工作空间 ID（英文字母、数字、下划线）"
                  value={newTenantId}
                  onChange={(e) => setNewTenantId(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="tenant-name">工作空间名称</Label>
                <Input
                  id="tenant-name"
                  placeholder="请输入工作空间名称"
                  value={newTenantName}
                  onChange={(e) => setNewTenantName(e.target.value)}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
                取消
              </Button>
              <Button onClick={handleCreateTenant} disabled={!newTenantId.trim() || !newTenantName.trim()}>
                创建
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
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
                  <Link href="/config/chatdb"><LassoSelectIcon /> ChatDB </Link>
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
