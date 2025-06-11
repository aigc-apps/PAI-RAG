"use client";

import React, { useState, useEffect } from "react";
import {
  TrashIcon,
  SettingsIcon,
  Edit,
  EyeIcon,
  EyeOffIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import * as Toast from "@radix-ui/react-toast";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Checkbox } from "@/components/ui/checkbox";
import { v4 as uuidv4 } from "uuid";

export const MaskedApiKey = ({ apiKey }: { apiKey: string }) => {
  const maskApiKey = (
    apiKey: string,
    prefixLength = 4,
    suffixLength = 3,
  ): string => {
    if (apiKey.length <= prefixLength + suffixLength) return apiKey; // 如果长度不够，直接返回原值
    return `${apiKey.slice(0, prefixLength)}*****${apiKey.slice(
      -suffixLength,
    )}`;
  };
  const [showFull, setShowFull] = useState(false);

  const toggleShow = () => setShowFull((prev) => !prev);

  return (
    <div className="flex items-center space-x-2">
      <span className="text-gray-700">
        {showFull ? apiKey : maskApiKey(apiKey)}
      </span>
      <button
        onClick={toggleShow}
        className="text-sm text-blue-500 hover:text-blue-700"
      >
        {showFull ? (
          <EyeOffIcon className="w-4 h-4" />
        ) : (
          <EyeIcon className="w-4 h-4" />
        )}
      </button>
    </div>
  );
};

class MCPConfig {
  id: string;
  name: string;
  url: string;
  type: string;
  active: boolean;

  constructor(
    id: string,
    name: string,
    url: string,
    type: string,
    active: boolean,
  ) {
    this.id = id;
    this.name = name;
    this.url = url;
    this.type = type;
    this.active = active;
  }
}

export default function McpConfig() {
  const [isOpen, setIsOpen] = useState(false); // 控制 AddMcpDialog 显示
  const [isEditOpen, setIsEditOpen] = useState(false); // 控制 EditMcpDialog 显示
  const [editingConfig, setEditingConfig] = useState<MCPConfig | null>(null);
  const [isLoading, setIsLoading] = useState(false); // 加载 AddMcpDialog 状态
  const [isEditLoading, setIsEditLoading] = useState(false); // 加载 EditMcpDialog 状态
  const [error, setError] = useState(""); // 错误信息
  const [toastState, setToastState] = useState({
    open: false,
    title: "",
    description: "",
    variant: "default" as "default" | "destructive",
  });

  const [addFormData, setAddFormData] = useState({
    id: uuidv4(),
    name: "未命名服务器",
    url: "",
    type: "sse",
    active: false,
  });

  const [mcpconfigs, setMcpConfigs] = useState(
    Array<{
      id: string;
      name: string;
      url: string;
      type: string;
      active: boolean;
    }>,
  ); // 存储 MCP 配置
  const [mcploading, setMcpLoading] = useState(true); // 加载状态
  const [mcperror, setMcpError] = useState(""); // 错误信息

  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8097;
        const res = await fetch(`http://localhost:${port}/v1/config/mcps`);
        if (!res.ok) throw new Error("获取配置失败");
        const data = await res.json();
        setMcpConfigs(data || []); // 更新状态
      } catch (err: any) {
        setMcpError(err || "加载失败");
      } finally {
        setMcpLoading(false);
      }
    };

    fetchConfigs();
  }, []);

  const handleEditClick = (config: MCPConfig) => {
    console.log("handleEditClick", config);
    setEditingConfig({ ...config }); // 深拷贝当前配置
    setIsEditOpen(true);
  };
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    const key = id.replace(/^mcp_/, "");
    setAddFormData((prev) => ({ ...prev, [key]: value }));
  };

  const handleEditInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    const key = id.replace(/^edit_mcp_/, "");
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

    const key = id.replace(/^edit_mcp_/, "");
    setEditingConfig((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        [key]: isChecked,
      };
    });
  };

  const addMCP = async () => {
    try {
      const mcp_data = {
        ...addFormData,
      };

      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
      const res = await fetch(`http://localhost:${port}/v1/config/mcps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(mcp_data), // 包装为数组
      });

      if (!res.ok) throw new Error("添加 MCP 配置失败");

      const mcpDto = await res.json();
      const newMcp = new MCPConfig(
        mcpDto.id,
        mcpDto.name,
        mcpDto.url,
        mcpDto.type,
        mcpDto.active,
      );
      setToastState({
        open: true,
        title: "MCP 配置已添加",
        description: "新模型配置已成功保存",
        variant: "default",
      });
      setIsOpen(false); // 关闭 AddDialog
      setMcpConfigs((prev) => [...prev, newMcp]); // 追加新 MCP 配置
    } catch (err: any) {
      setError(err || "添加失败，请重试"); // 显示错误信息
      setToastState({
        open: true,
        title: "添加失败",
        description: err.message || "请检查网络或重试",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const updatedMCP = async () => {
    try {
      if (!editingConfig) return;

      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8097;
      const res = await fetch(`http://localhost:${port}/v1/config/mcps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(editingConfig), // 包装为数组
      });

      if (!res.ok) throw new Error("修改 MCP 配置失败");
      setToastState({
        open: true,
        title: "MCP 配置已修改",
        description: "修改的模型配置已成功保存",
        variant: "default",
      });
      setIsEditOpen(false); // 关闭 EditDialog
      console.log("updateMCP", editingConfig);
      setMcpConfigs((prev) =>
        prev.map((config) =>
          config.id === editingConfig.id ? editingConfig : config,
        ),
      );
    } catch (err: any) {
      setError(err || "修改失败，请重试"); // 显示错误信息
      setToastState({
        open: true,
        title: "修改失败",
        description: err.message || "请检查网络或重试",
        variant: "destructive",
      });
    } finally {
      setIsEditLoading(false);
    }
  };
  const removeMCP = async (id: string) => {
    try {
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8097;
      const res = await fetch(`http://localhost:${port}/api/delete_mcp/${id}`, {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!res.ok) {
        throw new Error("删除失败，请检查网络或配置");
      }

      // 显示成功提示（可选）
      setToastState({
        open: true,
        title: "删除成功",
        description: "MCP 配置已移除",
        variant: "default",
      });

      // 删除成功后更新本地状态
      setMcpConfigs((prev) => prev.filter((config) => config.id !== id));
    } catch (err: any) {
      // 显示错误提示
      setToastState({
        open: true,
        title: "删除失败",
        description: err || "请稍后再试",
        variant: "destructive",
      });
    }
  };

  return (
    <div id="mcp">
      <div className={`transition-colors rounded-lg overflow-hidden`}>
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
                    <TableHead className="w-1/10">MCP ID</TableHead>
                    <TableHead className="w-1/10">MCP 名称</TableHead>
                    <TableHead className="w-2/5">MCP 链接</TableHead>
                    <TableHead className="w-1/10">MCP 类型</TableHead>
                    <TableHead className="w-1/10">是否激活</TableHead>
                    <TableHead className="w-1/5">操作</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {mcpconfigs.map((config) => (
                    <TableRow key={config.id}>
                      <TableCell>{config.id} </TableCell>
                      <TableCell>{config.name} </TableCell>
                      <TableCell>{config.url} </TableCell>
                      <TableCell> {config.type} </TableCell>
                      <TableCell>
                        <Checkbox id="terms" checked={config.active} />
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
                            )}{" "}
                            {/* 显示错误信息 */}
                            <DialogHeader>
                              <DialogTitle>编辑MCP配置</DialogTitle>
                              <DialogDescription>
                                编辑MCP配置信息后，点击保存。
                              </DialogDescription>
                            </DialogHeader>
                            <div className="grid gap-4 py-4">
                              <div className="grid grid-cols-4 items-center gap-4">
                                <Label
                                  htmlFor="edit_mcp_id"
                                  className="text-right"
                                >
                                  MCP ID
                                </Label>
                                <Input
                                  id="edit_mcp_id"
                                  defaultValue={editingConfig?.id || "null"}
                                  disabled
                                  className="col-span-3"
                                />
                              </div>
                              <div className="grid grid-cols-4 items-center gap-4">
                                <Label
                                  htmlFor="edit_mcp_name"
                                  className="text-right"
                                >
                                  MCP 名称
                                </Label>
                                <Input
                                  id="edit_mcp_name"
                                  defaultValue={editingConfig?.name || "null"}
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
                                  defaultValue={editingConfig?.url || "null"}
                                  onChange={handleEditInputChange}
                                  className="col-span-3"
                                />
                              </div>
                              <div className="grid grid-cols-4 items-center gap-4">
                                <Label
                                  htmlFor="edit_mcp_type"
                                  className="text-right"
                                >
                                  API Key
                                </Label>
                                <Input
                                  id="edit_mcp_type"
                                  defaultValue={editingConfig?.type || "null"}
                                  onChange={handleEditInputChange}
                                  className="col-span-3"
                                />
                              </div>
                              <div className="grid grid-cols-4 items-center gap-4">
                                <Label
                                  htmlFor="edit_mcp_active"
                                  className="text-right"
                                >
                                  是否激活
                                </Label>
                                <Checkbox
                                  id="edit_mcp_active"
                                  checked={editingConfig?.active || false}
                                  onCheckedChange={(checkedState) => {
                                    // 将 CheckedState 转换为 boolean
                                    const isChecked = checkedState === true;
                                    handleEditInputChangeCheckbox(
                                      isChecked,
                                      "edit_mcp_active",
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
                                {isEditLoading ? "提交中..." : "修改"}
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
              {error && <div className="text-red-500 mb-4">{error}</div>}{" "}
              {/* 显示错误信息 */}
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
                  </Label>
                  <Input
                    id="mcp_name"
                    placeholder="mcp_name"
                    onChange={handleInputChange}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="mcp_url" className="text-right">
                    MCP 链接
                  </Label>
                  <Input
                    id="mcp_url"
                    placeholder="mcp_url"
                    onChange={handleInputChange}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="mcp_type" className="text-right">
                    MCP 类型
                  </Label>
                  <Input
                    id="mcp_type"
                    placeholder="mcp_type"
                    onChange={handleInputChange}
                    className="col-span-3"
                  />
                </div>
              </div>
              <DialogFooter>
                <Button onClick={addMCP} type="submit" disabled={isLoading}>
                  {isLoading ? "提交中..." : "添加"}
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
              toastState.variant === "destructive"
                ? "border-red-500 bg-red-50 text-red-900"
                : "border-gray-200 bg-white text-gray-900"
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

          {/* 触发 Toast 的隐藏容器 */}
          <Toast.Viewport className="fixed bottom-0 right-0 z-[100] m-0 flex w-96 flex-col gap-2 p-6" />
        </div>
      </div>
    </div>
  );
}
