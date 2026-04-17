'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { TrashIcon, Edit, Plus, PlugZap, MoreHorizontal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { McpConfig } from './mcp';
import { toast } from 'sonner';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

export default function McpConfigPage() {
  const { t } = useI18n();

  const [isAddOpen, setIsAddOpen] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [editingConfig, setEditingConfig] = useState<McpConfig | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isEditLoading, setIsEditLoading] = useState(false);

  const emptyForm = {
    id: '',
    name: '',
    url: '',
    type: 'streamable_http',
    auth_token: '',
    need_token: false,
    enabled: true,
  };
  const [addFormData, setAddFormData] = useState(emptyForm);

  const [mcpconfigs, setMcpConfigs] = useState<
    Array<{
      id: string;
      name: string;
      url: string;
      type: string;
      auth_token: string;
      need_token: boolean;
      enabled: boolean;
    }>
  >([]);
  const [mcploading, setMcpLoading] = useState(true);
  const { tenantFetch } = useTenantFetch();

  const fetchConfigs = useCallback(async () => {
    try {
      setMcpLoading(true);
      const res = await tenantFetch(`/api/config/mcps`);
      if (!res.ok) throw new Error(t('config.loadError'));
      const data = await res.json();
      setMcpConfigs(data.data.items || []);
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setMcpLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchConfigs();
  }, [fetchConfigs]);

  const handleEditClick = (config: McpConfig) => {
    setEditingConfig({ ...config });
    setIsEditOpen(true);
  };

  const handleToggleEnabled = async (id: string, enabled: boolean) => {
    try {
      const res = await tenantFetch(`/api/config/mcps/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !enabled }),
      });
      if (!res.ok) throw new Error(t('config.mcp.updateStatusFailed'));
      setMcpConfigs((prev) =>
        prev.map((c) => (c.id === id ? { ...c, enabled: !enabled } : c)),
      );
      toast.success(t(!enabled ? 'config.mcp.mcpEnabled' : 'config.mcp.mcpDisabled'));
    } catch (err: any) {
      toast.error(`${t('config.mcp.updateFailed')}: ${err.message}`);
    }
  };

  const addMCP = async () => {
    try {
      setIsLoading(true);
      const mcp_data = {
        name: addFormData.name,
        url: addFormData.url,
        type: addFormData.type,
        auth_token: addFormData.auth_token,
        need_token: !!addFormData.auth_token,
        enabled: addFormData.enabled,
      };
      const res = await tenantFetch(`/api/config/mcps`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mcp_data),
      });
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`${t('config.mcp.addError')}: ${res.status} ${errorText || res.statusText}`);
      }
      await fetchConfigs();
      toast.success(t('config.mcp.addSuccess'));
      setIsAddOpen(false);
      setAddFormData(emptyForm);
    } catch (err: any) {
      toast.error(err.message || t('config.mcp.addFailed'));
    } finally {
      setIsLoading(false);
    }
  };

  const updatedMCP = async () => {
    try {
      if (!editingConfig) return;
      setIsEditLoading(true);
      const updateData: any = {
        name: editingConfig.name,
        url: editingConfig.url,
        type: editingConfig.type,
        enabled: editingConfig.enabled,
      };
      if (editingConfig.auth_token && editingConfig.auth_token.trim() !== '') {
        updateData.auth_token = editingConfig.auth_token;
        updateData.need_token = true;
      } else {
        updateData.need_token = false;
      }
      const res = await tenantFetch(`/api/config/mcps/${editingConfig.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updateData),
      });
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`${t('config.mcp.updateError')}: ${res.status} ${errorText || res.statusText}`);
      }
      await fetchConfigs();
      toast.success(t('config.mcp.updateSuccess'));
      setIsEditOpen(false);
    } catch (err: any) {
      toast.error(`${t('config.mcp.updateError')}${err.message || t('config.mcp.updateFailed2')}`);
    } finally {
      setIsEditLoading(false);
    }
  };

  const removeMCP = async (id: string) => {
    try {
      const res = await tenantFetch(`/api/config/mcps/${id}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!res.ok) throw new Error(t('config.mcp.deleteFailed'));
      toast.success(t('config.mcp.deleteSuccess'));
      setMcpConfigs((prev) => prev.filter((c) => c.id !== id));
    } catch (err: any) {
      toast.error(`${t('config.mcp.deleteError')}${err.message}`);
    }
  };

  return (
    <div id="mcp" className="settings-page">
      <div className="settings-page-header">
        <h1 className="page-title">
          MCP <span className="text-xs text-muted-foreground font-normal ml-1">· {mcpconfigs.length}</span>
        </h1>
        <Button size="sm" onClick={() => { setAddFormData(emptyForm); setIsAddOpen(true); }}>
          <Plus className="w-4 h-4" />
          {t('config.mcp.addMcp')}
        </Button>
      </div>

      <div className="settings-page-content">
        {mcploading ? (
          <div className="py-12 text-center">
            <p className="text-sm text-muted-foreground">{t('config.mcp.loading')}</p>
          </div>
        ) : mcpconfigs.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <PlugZap className="w-12 h-12 text-muted-foreground/40 mb-3" />
            <p className="text-sm text-muted-foreground mb-3">{t('config.mcp.noMcp')}</p>
            <Button size="sm" variant="outline" onClick={() => { setAddFormData(emptyForm); setIsAddOpen(true); }}>
              <Plus className="w-4 h-4" />
              {t('config.mcp.addMcp')}
            </Button>
          </div>
        ) : (
          <Table className="table-modern w-full border rounded-md overflow-hidden">
            <TableHeader>
              <TableRow>
                <TableHead className="w-[20%]">{t('config.mcp.mcpName')}</TableHead>
                <TableHead>{t('config.mcp.mcpUrl')}</TableHead>
                <TableHead className="w-[140px]">{t('config.mcp.mcpType')}</TableHead>
                <TableHead className="w-[90px] text-center">{t('config.mcp.isEnabled')}</TableHead>
                <TableHead className="w-[60px] text-right pr-4">{t('config.mcp.actions')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {mcpconfigs.map((config) => (
                <TableRow key={config.id}>
                  <TableCell className="font-medium text-sm">{config.name || ''}</TableCell>
                  <TableCell className="text-xs font-mono text-muted-foreground truncate max-w-0">
                    <span title={config.url}>{config.url || ''}</span>
                  </TableCell>
                  <TableCell>
                    <Badge className="badge-tech text-[10.5px] py-0">{config.type || ''}</Badge>
                  </TableCell>
                  <TableCell className="text-center">
                    <Switch
                      checked={config.enabled}
                      onCheckedChange={() => handleToggleEnabled(config.id, config.enabled)}
                    />
                  </TableCell>
                  <TableCell className="text-right pr-2">
                    <DropdownMenu modal={false}>
                      <DropdownMenuTrigger asChild>
                        <button type="button" className="icon-action-btn" aria-label="More">
                          <MoreHorizontal className="w-4 h-4" />
                        </button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end" className="menu-compact">
                        <DropdownMenuItem
                          onSelect={(e) => {
                            e.preventDefault();
                            setTimeout(() => handleEditClick(config), 0);
                          }}
                        >
                          <Edit />
                          {t('common.edit') || 'Edit'}
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          className="text-destructive focus:text-destructive"
                          onSelect={(e) => {
                            e.preventDefault();
                            setTimeout(() => removeMCP(config.id), 0);
                          }}
                        >
                          <TrashIcon />
                          {t('common.delete') || 'Delete'}
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>

      {/* Add Dialog */}
      <Dialog open={isAddOpen} onOpenChange={setIsAddOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <PlugZap className="w-4 h-4 text-primary" />
              {t('config.mcp.addMcp')}
            </DialogTitle>
            <DialogDescription>{t('config.mcp.addDialogDesc')}</DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-2">
            <div>
              <label htmlFor="mcp_name" className="form-label">
                {t('config.mcp.mcpNameLabel')}<span className="required">*</span>
              </label>
              <Input
                id="mcp_name"
                placeholder="MCP"
                value={addFormData.name}
                onChange={(e) => setAddFormData((p) => ({ ...p, name: e.target.value }))}
              />
            </div>
            <div>
              <label htmlFor="mcp_url" className="form-label">
                {t('config.mcp.mcpUrlLabel')}<span className="required">*</span>
              </label>
              <Input
                id="mcp_url"
                placeholder="URL"
                value={addFormData.url}
                onChange={(e) => setAddFormData((p) => ({ ...p, url: e.target.value }))}
              />
            </div>
            <div>
              <label htmlFor="mcp_type" className="form-label">
                {t('config.mcp.mcpTypeLabel')}<span className="required">*</span>
              </label>
              <Select
                value={addFormData.type}
                onValueChange={(v) => setAddFormData((p) => ({ ...p, type: v }))}
              >
                <SelectTrigger id="mcp_type">
                  <SelectValue placeholder={t('config.mcp.mcpTypePlaceholder')} />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="streamable_http">
                    <span suppressHydrationWarning>{t('config.mcp.streamableHttp')}</span>
                  </SelectItem>
                  <SelectItem value="sse">
                    <span suppressHydrationWarning>{t('config.mcp.sse')}</span>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <label htmlFor="mcp_auth_token" className="form-label">
                {t('config.mcp.bearerToken')}
              </label>
              <Input
                id="mcp_auth_token"
                type="password"
                placeholder={t('config.mcp.bearerTokenPlaceholder')}
                value={addFormData.auth_token}
                onChange={(e) => setAddFormData((p) => ({ ...p, auth_token: e.target.value }))}
              />
            </div>
            <div className="form-row-inline">
              <div className="form-row-inline-label">
                <span className="title">{t('config.mcp.defaultEnabled')}</span>
                <span className="hint">启用后可在对话中使用</span>
              </div>
              <Switch
                checked={addFormData.enabled}
                onCheckedChange={(v) => setAddFormData((p) => ({ ...p, enabled: v }))}
              />
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsAddOpen(false)}>
              {t('common.cancel') || 'Cancel'}
            </Button>
            <Button onClick={addMCP} disabled={isLoading}>
              {isLoading ? t('config.mcp.submitting') : t('config.mcp.addButton')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <PlugZap className="w-4 h-4 text-primary" />
              {t('config.mcp.editMcp')}
            </DialogTitle>
            <DialogDescription>{t('config.mcp.editDialogDesc')}</DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-2">
            <div>
              <label htmlFor="edit_mcp_name" className="form-label">
                {t('config.mcp.mcpNameLabel')}
              </label>
              <Input
                id="edit_mcp_name"
                value={editingConfig?.name || ''}
                onChange={(e) =>
                  setEditingConfig((prev) => (prev ? { ...prev, name: e.target.value } : prev))
                }
              />
            </div>
            <div>
              <label htmlFor="edit_mcp_url" className="form-label">
                {t('config.mcp.mcpUrlLabel')}
              </label>
              <Input
                id="edit_mcp_url"
                value={editingConfig?.url || ''}
                onChange={(e) =>
                  setEditingConfig((prev) => (prev ? { ...prev, url: e.target.value } : prev))
                }
              />
            </div>
            <div>
              <label htmlFor="edit_mcp_type" className="form-label">
                {t('config.mcp.mcpTypeLabel')}
              </label>
              <Select
                value={editingConfig?.type || 'streamable_http'}
                onValueChange={(v) =>
                  setEditingConfig((prev) => (prev ? { ...prev, type: v } : prev))
                }
              >
                <SelectTrigger id="edit_mcp_type">
                  <SelectValue placeholder={t('config.mcp.mcpTypePlaceholder')} />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="streamable_http">
                    <span suppressHydrationWarning>{t('config.mcp.streamableHttp')}</span>
                  </SelectItem>
                  <SelectItem value="sse">
                    <span suppressHydrationWarning>{t('config.mcp.sse')}</span>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <label htmlFor="edit_mcp_auth_token" className="form-label">
                {t('config.mcp.bearerToken')}
              </label>
              <Input
                id="edit_mcp_auth_token"
                type="password"
                value={editingConfig?.auth_token || ''}
                onChange={(e) =>
                  setEditingConfig((prev) =>
                    prev ? { ...prev, auth_token: e.target.value } : prev,
                  )
                }
              />
            </div>
            <div className="form-row-inline">
              <div className="form-row-inline-label">
                <span className="title">{t('config.mcp.isEnabled')}</span>
              </div>
              <Switch
                checked={editingConfig?.enabled || false}
                onCheckedChange={(v) =>
                  setEditingConfig((prev) => (prev ? { ...prev, enabled: v } : prev))
                }
              />
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsEditOpen(false)}>
              {t('common.cancel') || 'Cancel'}
            </Button>
            <Button onClick={updatedMCP} disabled={isEditLoading}>
              {isEditLoading ? t('config.mcp.submitting') : t('config.mcp.editButton')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
