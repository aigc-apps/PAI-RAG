'use client';
import { Search, Settings, Bot, Wrench, Database, PlugZap, SquareActivity, GlobeLock, ShieldCheck, Code, Users, Plus, Check, X, HardDrive } from 'lucide-react';
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
import { useI18n } from '@/app/providers/i18n';

export function AppSidebar() {
  const { tenantId, tenantName, tenants, setTenant, addTenant, removeTenant } = useTenant();
  const { t } = useI18n();
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
              <DropdownMenuLabel className="text-xs text-muted-foreground"><span suppressHydrationWarning>{t('workspace.select')}</span></DropdownMenuLabel>
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
                <span className="text-xs" suppressHydrationWarning>{t('workspace.create')}</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {/* 创建工作空间对话框 */}
        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle suppressHydrationWarning>{t('workspace.createTitle')}</DialogTitle>
              <DialogDescription suppressHydrationWarning>{t('workspace.createDescription')}</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="tenant-id" suppressHydrationWarning>{t('workspace.idLabel')}</Label>
                <Input
                  id="tenant-id"
                  placeholder={t('workspace.idPlaceholder')}
                  value={newTenantId}
                  onChange={(e) => setNewTenantId(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="tenant-name" suppressHydrationWarning>{t('workspace.nameLabel')}</Label>
                <Input
                  id="tenant-name"
                  placeholder={t('workspace.namePlaceholder')}
                  value={newTenantName}
                  onChange={(e) => setNewTenantName(e.target.value)}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
                <span suppressHydrationWarning>{t('common.cancel')}</span>
              </Button>
              <Button onClick={handleCreateTenant} disabled={!newTenantId.trim() || !newTenantName.trim()}>
                <span suppressHydrationWarning>{t('common.create')}</span>
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
                <Link href="/knowledgebases"><BookIcon /> <span suppressHydrationWarning>{t('sidebar.knowledgebase')}</span></Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <SidebarMenuButton asChild>
                <Link href="/apps"><AppWindowIcon /> <span suppressHydrationWarning>{t('sidebar.apps')}</span></Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <SidebarMenuButton asChild>
                <Link href="/evaluation"><Scale /> <span suppressHydrationWarning>{t('sidebar.evaluation')}</span></Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <CollapsibleTrigger asChild>
                <SidebarMenuButton>
                    <MessageCircle />
                    <span suppressHydrationWarning>{t('sidebar.conversation')}</span>
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
                  <Settings /> <span suppressHydrationWarning>{t('sidebar.settings')}</span>
                  <ChevronUp className="ml-auto" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent side="top" className="w-50">
                <DropdownMenuItem asChild>
                  <Link href="/config/model"><Bot /> <span suppressHydrationWarning>{t('sidebar.model')}</span></Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/vectordb"><Database /> <span suppressHydrationWarning>{t('sidebar.vectordb')}</span> </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/mcp"><PlugZap /> <span suppressHydrationWarning>{t('sidebar.mcp')}</span></Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/search"><Search /> <span suppressHydrationWarning>{t('sidebar.search')}</span></Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/code_sandbox"><Code /> <span suppressHydrationWarning>{t('sidebar.codeSandbox')}</span></Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/chatdb"><LassoSelectIcon /> <span suppressHydrationWarning>{t('sidebar.chatdb')}</span> </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/tracing"><SquareActivity /> <span suppressHydrationWarning>{t('sidebar.tracing')}</span></Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/role"><GlobeLock /> <span suppressHydrationWarning>{t('sidebar.role')}</span> </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/guardrail"><ShieldCheck /> <span suppressHydrationWarning>{t('sidebar.guardrail')}</span> </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/config/cache"><HardDrive /> <span suppressHydrationWarning>{t('sidebar.cache')}</span></Link>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  );
}
