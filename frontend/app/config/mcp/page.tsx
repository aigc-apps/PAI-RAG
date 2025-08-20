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
import * as Toast from '@radix-ui/react-toast';
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

export default function McpConfigPage() {
  const [isOpen, setIsOpen] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [editingConfig, setEditingConfig] = useState<McpConfig | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isEditLoading, setIsEditLoading] = useState(false);
  const [error, setError] = useState('');
  const [toastState, setToastState] = useState({
    open: false,
    title: '',
    description: '',
    variant: 'default' as 'default' | 'destructive',
  });

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
  const [mcperror, setMcpError] = useState('');

  // 提取 fetchConfigs 为可复用函数
  const fetchConfigs = useCallback(async () => {
    try {
      setMcpLoading(true);
      const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/v1/config/mcps`);
      if (!res.ok) throw new Error('获取配置失败');
      const data = await res.json();
      setMcpConfigs(data.data.items || []);
    } catch (err: any) {
      setMcpError(err || '加载失败');
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
      const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/v1/config/mcps/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !enabled }),
      });

      if (!res.ok) throw new Error('更新启用状态失败');

      // 直接更新本地状态，保持数据一致性
      setMcpConfigs((prev) =>
        prev.map((config) =>
          config.id === id ? { ...config, enabled: !enabled } : config
        )
      );

      setToastState({
        open: true,
        title: '状态已更新',
        description: `MCP 配置已${!enabled ? '启用' : '禁用'}`,
        variant: 'default',
      });
    } catch (err: any) {
      setToastState({
        open: true,
        title: '更新失败',
        description: err.message || '请检查网络或重试',
        variant: 'destructive',
      });
    }
  };

  const addMCP = async () => {
  try {
    setIsLoading(true);
    setError('');
    
    // 直接使用表单数据，不包含ID
    const mcp_data = {
      name: addFormData.name,
      url: addFormData.url,
      type: addFormData.type,
      auth_token: addFormData.auth_token,
      need_token: addFormData.auth_token ? true : false,
      enabled: addFormData.enabled,
    };
    
    const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/v1/config/mcps`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(mcp_data),
    });

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`添加 MCP 配置失败: ${res.status} ${errorText || res.statusText}`);
    }

    // 成功后重新拉取列表，确保ID一致性
    await fetchConfigs();
    
    setToastState({
      open: true,
      title: 'MCP 配置已添加',
      description: '新模型配置已成功保存',
      variant: 'default',
    });
    setIsOpen(false);

    // 重置表单数据
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
    const errorMessage = err.message || err.toString() || '添加失败，请重试';
    setError(errorMessage);
    setToastState({
      open: true,
      title: '添加失败',
      description: errorMessage,
      variant: 'destructive',
    });
  } finally {
    setIsLoading(false);
  }
};

const updatedMCP = async () => {
  try {
    if (!editingConfig) return;
    setIsEditLoading(true);
    setError('');
    
    // 构造干净的请求体
    const updateData: any = {
      name: editingConfig.name,
      url: editingConfig.url,
      type: editingConfig.type,
      enabled: editingConfig.enabled,
    };

    // 只有当 auth_token 不为空时才发送
    if (editingConfig.auth_token && editingConfig.auth_token.trim() !== '') {
      updateData.auth_token = editingConfig.auth_token;
      updateData.need_token = true;
    } else {
      updateData.need_token = false;
    }

    const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/v1/config/mcps/${editingConfig.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updateData),
    });

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`修改 MCP 配置失败: ${res.status} ${errorText || res.statusText}`);
    }
    
    // 成功后重新拉取列表
    await fetchConfigs();
    
    setToastState({
      open: true,
      title: 'MCP 配置已修改',
      description: '修改的模型配置已成功保存',
      variant: 'default',
    });
    setIsEditOpen(false);
  } catch (err: any) {
    const errorMessage = err.message || err.toString() || '修改失败，请重试';
    setError(errorMessage);
    setToastState({
      open: true,
      title: '修改失败',
      description: errorMessage,
      variant: 'destructive',
    });
  } finally {
    setIsEditLoading(false);
  }
};
  const removeMCP = async (id: string) => {
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/v1/config/mcps/${id}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!res.ok) {
        throw new Error('删除失败，请检查网络或配置');
      }

      setToastState({
        open: true,
        title: '删除成功',
        description: 'MCP 配置已移除',
        variant: 'default',
      });

      // 直接从本地状态中移除
      setMcpConfigs((prev) => prev.filter((config) => config.id !== id));
    } catch (err: any) {
      setToastState({
        open: true,
        title: '删除失败',
        description: err || '请稍后再试',
        variant: 'destructive',
      });
    }
  };

  return (
    <div id="mcp">
      <div className={'transition-colors rounded-lg overflow-hidden'}>
        <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-gray-200 rounded-xl bg-gray-50">
          {mcploading ? (
            <div className="py-12 text-center">
              <p className="text-gray-500">加载中...</p>
            </div>
          ) : error ? (
            <div className="py-12 text-center text-red-500">
              <p>{error}</p>
            </div>
          ) : mcpconfigs.length === 0 ? (
            <h3 className="text-lg font-medium text-gray-700 py-6">暂无 MCP</h3>
          ) : (
            <div className="gap-6 p-4 w-full">
              <Table className="w-full table-fixed border bg-white rounded-md overflow-hidden">
                <TableHeader className="bg-gray-100">
                  <TableRow>
                    <TableHead className="w-1/10">MCP 名称</TableHead>
                    <TableHead className="w-2/5">MCP 链接</TableHead>
                    <TableHead className="w-1/10">MCP 类型</TableHead>
                    <TableHead className="w-1/10">是否启用</TableHead>
                    <TableHead className="w-1/5">操作</TableHead>
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
                            {error && (
                              <div className="text-red-500 mb-4">{error}</div>
                            )}
                            <DialogHeader>
                              <DialogTitle>编辑MCP配置</DialogTitle>
                              <DialogDescription>
                                编辑MCP配置信息后，点击保存。
                              </DialogDescription>
                            </DialogHeader>
                            <div className="grid gap-4 py-4">
                              <div className="grid grid-cols-4 items-center gap-4">
                                <Label
                                  htmlFor="edit_mcp_name"
                                  className="text-right"
                                >
                                  MCP 名称
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
                                  MCP 链接
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
                                  MCP 类型
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
                                  Bear Token
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
                                  是否启用
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
                                {isEditLoading ? '提交中...' : '修改'}
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
          <p className="text-gray-500 mt-1">点击下方按钮添加新的 MCP</p>
          <Dialog open={isOpen} onOpenChange={setIsOpen}>
            <DialogTrigger asChild>
              <Button className="mt-4 px-4 py-2 text-white rounded-lg transition-colors">
                添加MCP
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[425px]">
              {error && <div className="text-red-500 mb-4">{error}</div>}
              <DialogHeader>
                <DialogTitle>添加MCP</DialogTitle>
                <DialogDescription>
                  填写MCP配置信息后，点击保存。
                </DialogDescription>
              </DialogHeader>
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="mcp_name" className="text-right">
                    MCP 名称
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
                    MCP 链接
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
                    MCP 类型
                    <span className="text-destructive">*</span>
                  </Label>
                  <Input
                    id="mcp_type"
                    placeholder="SSE / STDIO / HTTP"
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
                    Bearer Token
                  </Label>
                  <Input
                    id="mcp_auth_token"
                    type="password"
                    placeholder="Bearer Token (可选)"
                    value={addFormData.auth_token}
                    onChange={handleInputChange}
                    className="col-span-5"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="mcp_enabled" className="text-right">
                    默认启用
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
                  {isLoading ? '提交中...' : '添加'}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
          <Toast.Root
            open={toastState.open}
            onOpenChange={(open) =>
              setToastState((prev) => ({ ...prev, open }))
            }
            className={`grid grid-cols-[auto_1fr] items-center gap-x-4 rounded-md border px-4 py-6 shadow-lg transition-all data-[state=open]:animate-slideIn data-[state=closed]:animate-fadeOut ${
              toastState.variant === 'destructive'
                ? 'border-red-500 bg-red-50 text-red-900'
                : 'border-gray-200 bg-white text-gray-900'
            }`}
          >
            <Toast.Description className="pl-4 text-sm font-medium">
              {toastState.description}
            </Toast.Description>
            <Toast.Action
              altText="关闭"
              onClick={() =>
                setToastState((prev) => ({ ...prev, open: false }))
              }
            >
              ×
            </Toast.Action>
          </Toast.Root>

          <Toast.Viewport className="fixed bottom-0 right-0 z-[100] m-0 flex w-96 flex-col gap-2 p-6" />
        </div>
      </div>
    </div>
  );
}
