"use client"

import React, {useState, useEffect} from "react";
import {
  TrashIcon,
  SettingsIcon,
  Edit,
  EyeIcon, 
  EyeOffIcon
} from "lucide-react";
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import * as Toast from "@radix-ui/react-toast"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

export const MaskedApiKey = ({ apiKey }: { apiKey: string }) => {
  const maskApiKey = (apiKey: string, prefixLength = 4, suffixLength = 3): string => {
    if (apiKey.length <= prefixLength + suffixLength) return apiKey; // 如果长度不够，直接返回原值
    return `${apiKey.slice(0, prefixLength)}*****${apiKey.slice(-suffixLength)}`;
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
        className="text-sm text-black-500 hover:text-black-700"
      >
        {showFull ? <EyeOffIcon className="w-4 h-4" /> : <EyeIcon className="w-4 h-4" />}
      </button>
    </div>
  );
};

class LLMConfig {
  id: number;
  source: string;
  model_name: string;
  api_key: string;
  max_context: number;

  constructor(id: number, source: string, model_name: string, api_key: string, max_context: number) {
    this.id = id;
    this.source = source;
    this.model_name = model_name;
    this.api_key = api_key;
    this.max_context = max_context;
  }
}

export default function LlmConfig() {
  const [isOpen, setIsOpen] = useState(false); // 控制 AddLlmDialog 显示
  const [isEditOpen, setIsEditOpen] = useState(false); // 控制 EditLlmDialog 显示
  const [editingConfig, setEditingConfig] = useState<LLMConfig | null>(null);
  const [isLoading, setIsLoading] = useState(false); // 加载 AddLlmDialog 状态
  const [isEditLoading, setIsEditLoading] = useState(false); // 加载 EditLlmDialog 状态
  const [error, setError] = useState(""); // 错误信息
  const [toastState, setToastState] = useState({
    open: false,
    title: "",
    description: "",
    variant: "default" as "default" | "destructive",
  });
  
  const [addFormData, setAddFormData] = useState({
    id: Date.now(),
    model_name: "qwen-max",
    source: "qwen",
    api_key: "sk-xxxxxx"
  });

  const [editFormData, setEditFormData] = useState({
    id: Date.now(),
    model_name: "qwen-max",
    source: "qwen",
    api_key: "sk-xxxxxx"
  });

  const [llmconfigs, setLlmConfigs] = useState(Array<{
      id: number;
      source: string;
      model_name: string;
      api_key: string;
      max_context: number;
  }>); // 存储 LLM 配置
  const [llmloading, setLlmLoading] = useState(true); // 加载状态
  const [llmerror, setLlmError] = useState(""); // 错误信息
  
  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8097;
        const res = await fetch(`http://localhost:${port}/api/configs`);
        if (!res.ok) throw new Error("获取配置失败");
        const data = await res.json();
        setLlmConfigs(data.llm_config || []); // 更新状态
      } catch (err: any) {
        setLlmError(err || "加载失败");
      } finally {
        setLlmLoading(false);
      }
    };

    fetchConfigs();
  }, []);

  const handleEditClick = (config: LLMConfig) => {
    console.log("handleEditClick", config);
    setEditingConfig({ ...config }); // 深拷贝当前配置
    setIsEditOpen(true);
  };
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    setAddFormData((prev) => ({ ...prev, [id]: value }));
  };

  const handleEditInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    console.log("formData:", id , value)
    const key = id.replace(/^edit_/, '');
    setEditFormData((prev) => ({ ...prev, [key]: value }));
    console.log("更新后的 formData:", { ...editFormData, [key]: value });
    setEditingConfig((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        [key]: value,
      };
    });
  };
  
  const addLLM = async () => {
    try {
      const newLLM = {
        ...addFormData,
        max_context: 0 // 默认值
      };
  
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8097;
      const res = await fetch(`http://localhost:${port}/api/add_llm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ llm_config: newLLM }) // 包装为数组
      });
  
      if (!res.ok) throw new Error("添加 LLM 配置失败");
      setToastState({
        open: true,
        title: "LLM 配置已添加",
        description: "新模型配置已成功保存",
        variant: "default",
      });
      setIsOpen(false); // 关闭 AddDialog
      setLlmConfigs((prev) => [...prev, newLLM]); // 追加新 LLM 配置
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

  const updatedLLM = async () => {
    try {
      if (!editingConfig) return;
  
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8097;
      const res = await fetch(`http://localhost:${port}/api/add_llm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ llm_config: editingConfig }) // 包装为数组
      });
  
      if (!res.ok) throw new Error("修改 LLM 配置失败");
      setToastState({
        open: true,
        title: "LLM 配置已修改",
        description: "修改的模型配置已成功保存",
        variant: "default",
      });
      setIsEditOpen(false); // 关闭 EditDialog
      console.log("updateLLM", editingConfig)
      setLlmConfigs((prev) =>
        prev.map((config) =>
          config.id === editingConfig.id ? editingConfig : config
        )
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
  }
  const removeLLM = async (id: number) => {
    try {
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8097;
      const res = await fetch(`http://localhost:${port}/api/delete_llm/${id}`, {
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
        description: "LLM 配置已移除",
        variant: "default",
      });

      // 删除成功后更新本地状态
      setLlmConfigs((prev) => prev.filter((config) => config.id !== id));
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

  return <div id="llm">
      <div className={`transition-colors rounded-lg overflow-hidden`}>
        <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-gray-200 rounded-xl bg-gray-50">
          
          {llmloading ? (
            <div className="py-12 text-center">
              <p className="text-gray-500">加载中...</p>
            </div>
          ) : error ? (
            <div className="py-12 text-center text-red-500">
              <p>{error}</p>
            </div>
          ) : llmconfigs.length === 0 ? (
            <h3 className="text-lg font-medium text-gray-700 py-6">暂无 LLM</h3>
          ) : (
            <div className="gap-6 p-4 w-full">
              <Table className="w-full table-fixed border bg-white rounded-md overflow-hidden">
                <TableHeader className="bg-gray-100">
                  <TableRow>
                    <TableHead className="w-1/5">模型ID</TableHead>
                    <TableHead className="w-1/5">模型名称</TableHead>
                    <TableHead className="w-1/5">模型来源</TableHead>
                    <TableHead className="w-1/5">API Key</TableHead>
                    <TableHead className="w-1/5">操作</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                    {llmconfigs.map((config) => (
                    <TableRow key={config.id}>
                      <TableCell>{config.id} </TableCell>
                      <TableCell>{config.model_name} </TableCell>
                      <TableCell>{config.source} </TableCell>
                      <TableCell> <MaskedApiKey apiKey={config.api_key} /> </TableCell>
                      <TableCell>
                        <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
                          <DialogTrigger asChild>
                            <button className="text-black-500 hover:text-black-700 px-1 py-1" onClick={() => handleEditClick(config)}>
                              <Edit className="w-4 h-4" />
                            </button>
                          </DialogTrigger>
                          <DialogContent className="sm:max-w-[425px]">
                            {error && <div className="text-red-500 mb-4">{error}</div>} {/* 显示错误信息 */}
                            <DialogHeader>
                              <DialogTitle>编辑模型配置</DialogTitle>
                              <DialogDescription>
                                编辑模型配置信息后，点击保存。
                              </DialogDescription>
                            </DialogHeader>
                            <div className="grid gap-4 py-4">
                              <div className="grid grid-cols-4 items-center gap-4">
                                <Label htmlFor="edit_id" className="text-right">
                                  模型ID
                                </Label>
                                <Input id="edit_id" defaultValue={editingConfig?.id || "null"} disabled
                                  className="col-span-3" />
                              </div>
                              <div className="grid grid-cols-4 items-center gap-4">
                                <Label htmlFor="edit_model_name" className="text-right">
                                  模型名称
                                </Label>
                                <Input id="edit_model_name" defaultValue={editingConfig?.model_name || "null"}
                                  onChange={handleEditInputChange} className="col-span-3" />
                              </div>
                              <div className="grid grid-cols-4 items-center gap-4">
                                <Label htmlFor="edit_source" className="text-right">
                                  模型来源
                                </Label>
                                <Input id="edit_source" defaultValue={editingConfig?.source || "null"}
                                  onChange={handleEditInputChange} className="col-span-3" />
                              </div>
                              <div className="grid grid-cols-4 items-center gap-4">
                                <Label htmlFor="edit_api_key" className="text-right">
                                  API Key
                                </Label>
                                <Input id="edit_api_key" defaultValue={editingConfig?.api_key || "null"}
                                  onChange={handleEditInputChange} className="col-span-3" />
                              </div>
                            </div>
                            <DialogFooter>
                              <Button onClick={updatedLLM} type="submit" disabled={isEditLoading}>
                                {isEditLoading ? "提交中..." : "修改"}
                              </Button>
                            </DialogFooter>
                          </DialogContent>
                        </Dialog>
                        
                        <button
                            onClick={() => removeLLM(config.id)}
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
          <p className="text-gray-500 mt-1">
            点击下方按钮添加新的 LLM
          </p>
          <Dialog open={isOpen} onOpenChange={setIsOpen}>
            <DialogTrigger asChild>
              <Button className="mt-4 px-4 py-2 text-white rounded-lg transition-colors">
              添加LLM
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[425px]">
              {error && <div className="text-red-500 mb-4">{error}</div>} {/* 显示错误信息 */}
              <DialogHeader>
                <DialogTitle>添加大模型</DialogTitle>
                <DialogDescription>
                  填写模型配置信息后，点击保存。
                </DialogDescription>
              </DialogHeader>
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="model_name" className="text-right">
                    模型名称
                  </Label>
                  <Input id="model_name" placeholder="model_name"
                    onChange={handleInputChange} className="col-span-3" />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="source" className="text-right">
                    模型来源
                  </Label>
                  <Input id="source" placeholder="source"
                    onChange={handleInputChange} className="col-span-3" />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="api_key" className="text-right">
                    API Key
                  </Label>
                  <Input id="api_key" placeholder="api_key"
                     onChange={handleInputChange} className="col-span-3" />
                </div>
              </div>
              <DialogFooter>
                <Button onClick={addLLM} type="submit" disabled={isLoading}>
                  {isLoading ? "提交中..." : "添加"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
          <Toast.Root
            open={toastState.open}
            onOpenChange={(open) => setToastState((prev) => ({ ...prev, open }))}
            className={`grid grid-cols-[auto_1fr] items-center gap-x-4 rounded-md border px-4 py-6 shadow-lg transition-all data-[state=open]:animate-slideIn data-[state=closed]:animate-fadeOut ${
              toastState.variant === "destructive"
                ? "border-red-500 bg-red-50 text-red-900"
                : "border-gray-200 bg-white text-gray-900"
            }`}
          >
            <Toast.Description className="pl-4 text-sm font-medium">{toastState.description}</Toast.Description>
            <Toast.Action altText="关闭" onClick={() => setToastState((prev) => ({ ...prev, open: false }))}>
              ×
            </Toast.Action>
          </Toast.Root>

          {/* 触发 Toast 的隐藏容器 */}
          <Toast.Viewport className="fixed bottom-0 right-0 z-[100] m-0 flex w-96 flex-col gap-2 p-6" />
        </div>
      </div>
  </div>;
}
