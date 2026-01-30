'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  TrashIcon,
  SettingsIcon,
  Edit,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Checkbox } from '@/components/ui/checkbox';
import { v4 as uuidv4 } from 'uuid';
import { McpConfig } from './mcp';
import { toast } from 'sonner';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

export default function McpConfigPage() {
  const { t } = useI18n();

  const [isOpen, setIsOpen] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [editingConfig, setEditingConfig] = useState<McpConfig | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isEditLoading, setIsEditLoading] = useState(false);

  const [addFormData, setAddFormData] = useState({
    id: '',
    name: '',
    url: '',
    type: '',
    auth_token: '',
    need_token: false,
    enabled: true,
  });

  const [mcpconfigs, setMcpConfigs] = useState<Array<{
    id: string;
    name: string;
    url: string;
    type: string;
    auth_token: string;
    need_token: boolean;
    enabled: boolean;
  }>>([]);
  const [mcploading, setMcpLoading] = useState(true);
  const { tenantFetch } = useTenantFetch();
  // 提取 fetchConfigs 为可复用函数
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

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    const key = id.replace(/^mcp_/, '');
    setAddFormData((prev) => ({ ...prev, [key]: value }));
  };

  const handleEditInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    const key = id.replace(/^edit_mcp_/, '');
    setEditingConfig((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        [key]: value,
      };
    });
  };

  const handleEditInputChangeCheckbox = (isChecked: boolean, id?: string) => {
    if (!id || !editingConfig) return;

    const key = id.replace(/^edit_mcp_/, '');
    setEditingConfig((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        [key]: isChecked,
      };
    });
  };

  const handleToggleEnabled = async (id: string, enabled: boolean) => {
    try {
      const res = await tenantFetch(`/api/config/mcps/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !enabled }),
      });

      if (!res.ok) throw new Error(t('config.mcp.updateStatusFailed'));

      // Directly update local state to maintain data consistency
      setMcpConfigs((prev) =>
        prev.map((config) =>
          config.id === id ? { ...config, enabled: !enabled } : config
        )
      );

      toast.success(t(!enabled ? 'config.mcp.mcpEnabled' : 'config.mcp.mcpDisabled'));
    } catch (err: any) {
      toast.error(`${t('config.mcp.updateFailed')}: ${err.message}`);
    }
  };

  const addMCP = async () => {
  try {
    setIsLoading(true);
    
    // Use form data directly, without ID
    const mcp_data = {
      name: addFormData.name,
      url: addFormData.url,
      type: addFormData.type,
      auth_token: addFormData.auth_token,
      need_token: addFormData.auth_token ? true : false,
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

    // After success, re-fetch list to ensure ID consistency
    await fetchConfigs();
    toast.success(t('config.mcp.addSuccess'));

    setIsOpen(false);

    // Reset form data
    setAddFormData({
      id: uuidv4(),
      name: '',
      url: '',
      type: 'sse',
      auth_token: '',
      need_token: false,
      enabled: true,
    });
  } catch (err: any) {
    const errorMessage = err.message || err.toString() || t('config.mcp.addFailed');
    toast.error(errorMessage);
  } finally {
    setIsLoading(false);
  }
};

const updatedMCP = async () => {
  try {
    if (!editingConfig) return;
    setIsEditLoading(true);
    
    // Construct clean request body
    const updateData: any = {
      name: editingConfig.name,
      url: editingConfig.url,
      type: editingConfig.type,
      enabled: editingConfig.enabled,
    };

    // Only send auth_token if it's not empty
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
    
    // After success, re-fetch list
    await fetchConfigs();
    
    toast.success(t('config.mcp.updateSuccess'));
    setIsEditOpen(false);
  } catch (err: any) {
    const errorMessage = err.message || err.toString() || t('config.mcp.updateFailed2');
    toast.error(`${t('config.mcp.updateError')}${errorMessage}`);
  } finally {
    setIsEditLoading(false);
  }
};
  const removeMCP = async (id: string) => {
    try {
      const res = await tenantFetch(`/api/config/mcps/${id}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!res.ok) {
        throw new Error(t('config.mcp.deleteFailed'));
      }
      
      toast.success(t('config.mcp.deleteSuccess'));


      // Remove directly from local state
      setMcpConfigs((prev) => prev.filter((config) => config.id !== id));
    } catch (err: any) {
      toast.error(`${t('config.mcp.deleteError')}${err.message}`);
    }
  };

  return (
    <div id="mcp">
      <div className={'transition-colors rounded-lg overflow-hidden'}>
        <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-gray-200 rounded-xl bg-gray-50">
          {mcploading ? (
            <div className="py-12 text-center">
              <p className="text-gray-500">{t('config.mcp.loading')}</p>
            </div>
          ) : mcpconfigs.length === 0 ? (
            <h3 className="text-lg font-medium text-gray-700 py-6">{t('config.mcp.noMcp')}</h3>
          ) : (
            <div className="gap-6 p-4 w-full">
              <Table className="w-full table-fixed border bg-white rounded-md overflow-hidden">
                <TableHeader className="bg-gray-100">
                  <TableRow>
                    <TableHead className="w-1/10">{t('config.mcp.mcpName')}</TableHead>
                    <TableHead className="w-2/5">{t('config.mcp.mcpUrl')}</TableHead>
                    <TableHead className="w-1/10">{t('config.mcp.mcpType')}</TableHead>
                    <TableHead className="w-1/10">{t('config.mcp.isEnabled')}</TableHead>
                    <TableHead className="w-1/5">{t('config.mcp.actions')}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {mcpconfigs.map((config) => (
                    <TableRow key={config.id}>
                      <TableCell>{config.name || ''}</TableCell>
                      <TableCell>{config.url || ''}</TableCell>
                      <TableCell>{config.type || ''}</TableCell>
                      <TableCell>
                        <Checkbox
                          id={`enabled-${config.id}`}
                          checked={config.enabled}
                          onCheckedChange={(checked) => 
                            handleToggleEnabled(config.id, config.enabled)
                          }
                        />
                      </TableCell>
                      <TableCell>
                        <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
                          <DialogTrigger asChild>
                            <button
                              className="text-black-500 hover:text-black-700 px-1 py-1"
                              onClick={() => handleEditClick(config)}
                            >
                              <Edit className="w-4 h-4" />
                            </button>
                          </DialogTrigger>
                          <DialogContent className="sm:max-w-[425px]">
                            <DialogHeader>
                              <DialogTitle>{t('config.mcp.editMcp')}</DialogTitle>
                              <DialogDescription>
                                {t('config.mcp.editDialogDesc')}
                              </DialogDescription>
                            </DialogHeader>
                            <div className="grid gap-4 py-4">
                              <div className="grid grid-cols-4 items-center gap-4">
                                <Label
                                  htmlFor="edit_mcp_name"
                                  className="text-right"
                                >
                                  {t('config.mcp.mcpNameLabel')}
                                </Label>
                                <Input
                                  id="edit_mcp_name"
                                  value={editingConfig?.name || ''}
                                  onChange={handleEditInputChange}
                                  className="col-span-3"
                                />
                              </div>
                              <div className="grid grid-cols-4 items-center gap-4">
                                <Label
                                  htmlFor="edit_mcp_url"
                                  className="text-right"
                                >
                                  {t('config.mcp.mcpUrlLabel')}
                                </Label>
                                <Input
                                  id="edit_mcp_url"
                                  value={editingConfig?.url || ''}
                                  onChange={handleEditInputChange}
                                  className="col-span-3"
                                />
                              </div>
                              <div className="grid grid-cols-4 items-center gap-4">
                                <Label
                                  htmlFor="edit_mcp_type"
                                  className="text-right"
                                >
                                  {t('config.mcp.mcpTypeLabel')}
                                </Label>
                                <Input
                                  id="edit_mcp_type"
                                  value={editingConfig?.type || ''}
                                  onChange={handleEditInputChange}
                                  className="col-span-3"
                                />
                              </div>
                              <div className="grid grid-cols-7 items-center gap-4">
                                <Label
                                  htmlFor="edit_mcp_auth_token"
                                  className="text-right col-span-2"
                                >
                                  {t('config.mcp.bearerToken')}
                                </Label>
                                <Input
                                  id="edit_mcp_auth_token"
                                  type="password"
                                  value={
                                    editingConfig?.auth_token || ''
                                  }
                                  onChange={handleEditInputChange}
                                  className="col-span-5"
                                />
                              </div>
                              <div className="grid grid-cols-4 items-center gap-4">
                                <Label
                                  htmlFor="edit_mcp_enabled"
                                  className="text-right"
                                >
                                  {t('config.mcp.isEnabled')}
                                </Label>
                                <Checkbox
                                  id="edit_mcp_enabled"
                                  checked={editingConfig?.enabled || false}
                                  onCheckedChange={(checkedState) => {
                                    const isChecked = checkedState === true;
                                    handleEditInputChangeCheckbox(
                                      isChecked,
                                      'edit_mcp_enabled',
                                    );
                                  }}
                                  className="col-span-3"
                                />
                              </div>
                            </div>
                            <DialogFooter>
                              <Button
                                onClick={updatedMCP}
                                type="submit"
                                disabled={isEditLoading}
                              >
                                {isEditLoading ? t('config.mcp.submitting') : t('config.mcp.editButton')}
                              </Button>
                            </DialogFooter>
                          </DialogContent>
                        </Dialog>

                        <button
                          onClick={() => removeMCP(config.id)}
                          className="text-red-500 hover:text-red-700 px-4 py-1"
                        >
                          <TrashIcon className="w-4 h-4" />
                        </button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
          <SettingsIcon className="w-10 h-6 text-gray-400 mb-4" />
          <p className="text-gray-500 mt-1">{t('config.mcp.clickToAddNew')}</p>
          <Dialog open={isOpen} onOpenChange={setIsOpen}>
            <DialogTrigger asChild>
              <Button className="mt-4 px-4 py-2 text-white rounded-lg transition-colors">
                {t('config.mcp.addMcp')}
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[425px]">
              <DialogHeader>
                <DialogTitle>{t('config.mcp.addMcp')}</DialogTitle>
                <DialogDescription>
                  {t('config.mcp.addDialogDesc')}
                </DialogDescription>
              </DialogHeader>
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="mcp_name" className="text-right">
                    {t('config.mcp.mcpNameLabel')}
                    <span className="text-destructive">*</span>
                  </Label>
                  <Input
                    id="mcp_name"
                    placeholder="MCP"
                    value={addFormData.name}
                    onChange={handleInputChange}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="mcp_url" className="text-right">
                    {t('config.mcp.mcpUrlLabel')}
                    <span className="text-destructive">*</span>
                  </Label>
                  <Input
                    id="mcp_url"
                    placeholder="URL"
                    value={addFormData.url}
                    onChange={handleInputChange}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="mcp_type" className="text-right">
                    {t('config.mcp.mcpTypeLabel')}
                    <span className="text-destructive">*</span>
                  </Label>
                  <Input
                    id="mcp_type"
                    placeholder={t('config.mcp.mcpTypePlaceholder')}
                    value={addFormData.type}
                    onChange={handleInputChange}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-7 items-center gap-4">
                  <Label
                    htmlFor="mcp_auth_token"
                    className="text-right col-span-2"
                  >
                    {t('config.mcp.bearerToken')}
                  </Label>
                  <Input
                    id="mcp_auth_token"
                    type="password"
                    placeholder={t('config.mcp.bearerTokenPlaceholder')}
                    value={addFormData.auth_token}
                    onChange={handleInputChange}
                    className="col-span-5"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="mcp_enabled" className="text-right">
                    {t('config.mcp.defaultEnabled')}
                  </Label>
                  <Checkbox
                    id="mcp_enabled"
                    checked={addFormData.enabled}
                    onCheckedChange={(checked) => 
                      setAddFormData(prev => ({ ...prev, enabled: checked === true }))
                    }
                    className="col-span-3"
                  />
                </div>
              </div>
              <DialogFooter>
                <Button onClick={addMCP} type="submit" disabled={isLoading}>
                  {isLoading ? t('config.mcp.submitting') : t('config.mcp.addButton')}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>
    </div>
  );
}
