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
import { Switch } from "@/components/ui/switch";
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
        className="text-sm text-black-500 hover:text-black-700"
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

class LLMConfig {
  id: string;
  model_id: string;
  source: string;
  model: string;
  api_key: string;
  base_url: string;
  max_context: number;
  enabled: boolean = true;

  constructor(
    id: string,
    model_id: string,
    source: string,
    model: string,
    api_key: string,
    base_url: string,
    max_context: number,
    enabled: boolean,
  ) {
    this.id = id;
    this.model_id = model_id;
    this.source = source;
    this.model = model;
    this.api_key = api_key;
    this.base_url = base_url;
    this.max_context = max_context;
    this.enabled = enabled;
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
    id: uuidv4(),
    model: "qwen-max",
    model_id: "qwen-max",
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key: "sk-xxxxxx",
    enabled: true,
  });

  const [editFormData, setEditFormData] = useState({
    id: uuidv4(),
    model_id: "qwen-max",
    model: "qwen-max",
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key: "sk-xxxxxx",
    enabled: true,
  });

  const [llmconfigs, setLlmConfigs] = useState(
    Array<{
      id: string;
      source: string;
      model: string;
      model_id: string;
      api_key: string;
      base_url: string;
      max_context: number;
      enabled: boolean;
    }>,
  ); // 存储 LLM 配置
  const [llmloading, setLlmLoading] = useState(true); // 加载状态
  const [llmerror, setLlmError] = useState(""); // 错误信息

  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
        const res = await fetch(`http://localhost:${port}/v1/config/llms`);
        if (!res.ok) throw new Error("获取配置失败");
        const data = await res.json();
        setLlmConfigs(data || []); // 更新状态
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
    console.log("formData:", id, value);
    const key = id.replace(/^edit_/, "");
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
        max_context: 0,
        id: uuidv4(),
      };

      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
      const res = await fetch(`http://localhost:${port}/v1/config/llms`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newLLM), // 包装为数组
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

  const updatedLLM = async (id: string) => {
    try {
      if (!editingConfig) return;

      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
      const res = await fetch(`http://localhost:${port}/v1/config/llms/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(editingConfig), // 包装为数组
      });

      if (!res.ok) throw new Error("修改 LLM 配置失败");
      setToastState({
        open: true,
        title: "LLM 配置已修改",
        description: "修改的模型配置已成功保存",
        variant: "default",
      });
      setIsEditOpen(false); // 关闭 EditDialog
      console.log("updateLLM", editingConfig);
      setLlmConfigs((prev) =>
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
  const removeLLM = async (id: string) => {
    try {
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
      const res = await fetch(`http://localhost:${port}/v1/config/llms/${id}`, {
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

  const handleActivateToggle = async (config: LLMConfig) => {
    console.log("handleActivateToggle", config);
    const updatedConfig = {
      ...config,
      enabled: !config.enabled,
    };
    console.log("updatedConfig", updatedConfig);
    // 更新本地状态

    try {
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8097;
      const res = await fetch(`http://localhost:${port}/api/add_llm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ llm_config: updatedConfig }), // 包装为数组
      });

      if (!res.ok) throw new Error("更新 LLM 状态失败");
      setToastState({
        open: true,
        title: "LLM 状态已更新",
        description: "LLM 状态已更新成功",
        variant: "default",
      });

      setLlmConfigs((prev) =>
        prev.map((item) => (item.id === config.id ? updatedConfig : item)),
      );
    } catch (err: any) {
      setError(err || "修改失败，请重试"); // 显示错误信息
      setToastState({
        open: true,
        title: "修改失败",
        description: err.message || "请检查网络或重试",
        variant: "destructive",
      });
    }
  };

  return (
    <div id="llm">
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
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {llmconfigs.map((config) => (
                  <div
                    key={config.id}
                    className="bg-white rounded-lg shadow-sm border border-gray-100 p-4 hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-center gap-2 mb-3">
                      <h3 className="font-medium text-gray-800">
                        {config.model}
                      </h3>
                      <span className="text-xs bg-blue-50 text-blue-500 px-2 py-0.5 rounded-full">
                        {config.source}
                      </span>
                    </div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-600">{config.model_id}</p>
                      <p className="text-sm font-mono bg-gray-50 px-2 py-1 rounded text-gray-800 truncate">
                        {config.base_url}
                      </p>
                    </div>
                    <div className="flex justify-end gap-2 pt-2 border-t border-gray-100">
                      <Switch
                        checked={config.enabled}
                        onCheckedChange={() => handleActivateToggle(config)}
                        className="ml-auto"
                      />
                      <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
                        <DialogTrigger asChild>
                          <button
                            className="text-black-100 hover:text-black-100 px-1 py-1"
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
                            <DialogTitle>编辑模型配置</DialogTitle>
                            <DialogDescription>
                              编辑模型配置信息后，点击保存。
                            </DialogDescription>
                          </DialogHeader>
                          <div className="grid gap-4 py-4">
                            <div className="grid grid-cols-4 items-center gap-4">
                              <Label
                                htmlFor="edit_model_id"
                                className="text-right"
                              >
                                模型ID
                              </Label>
                              <Input
                                id="edit_model_id"
                                defaultValue={editingConfig?.model_id || "null"}
                                disabled
                                className="col-span-3"
                              />
                            </div>
                            <div className="grid grid-cols-4 items-center gap-4">
                              <Label
                                htmlFor="edit_model_name"
                                className="text-right"
                              >
                                模型名称
                              </Label>
                              <Input
                                id="edit_model_name"
                                defaultValue={
                                  editingConfig?.model_name || "null"
                                }
                                onChange={handleEditInputChange}
                                className="col-span-3"
                              />
                            </div>
                            <div className="grid grid-cols-4 items-center gap-4">
                              <Label
                                htmlFor="edit_source"
                                className="text-right"
                              >
                                模型来源
                              </Label>
                              <Input
                                id="edit_source"
                                defaultValue={editingConfig?.base_url || "null"}
                                onChange={handleEditInputChange}
                                className="col-span-3"
                              />
                            </div>
                            <div className="grid grid-cols-4 items-center gap-4">
                              <Label
                                htmlFor="edit_api_key"
                                className="text-right"
                              >
                                API Key
                              </Label>
                              <Input
                                id="edit_api_key"
                                defaultValue={editingConfig?.api_key || "null"}
                                onChange={handleEditInputChange}
                                className="col-span-3"
                              />
                            </div>
                          </div>
                          <DialogFooter>
                            <Button
                              onClick={updatedLLM}
                              type="submit"
                              disabled={isEditLoading}
                            >
                              {isEditLoading ? "提交中..." : "修改"}
                            </Button>
                          </DialogFooter>
                        </DialogContent>
                      </Dialog>
                      <button
                        onClick={() => removeLLM(config.id)}
                        className="text-red-500 hover:text-red-700 p-1"
                        aria-label="删除"
                      >
                        <TrashIcon className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          <SettingsIcon className="w-10 h-6 text-gray-400 mb-4" />
          <p className="text-gray-500 mt-1">点击下方按钮添加新的 LLM</p>
          <Dialog open={isOpen} onOpenChange={setIsOpen}>
            <DialogTrigger asChild>
              <Button className="mt-4 px-4 py-2 text-white rounded-lg transition-colors">
                添加LLM
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[425px]">
              {error && <div className="text-red-500 mb-4">{error}</div>}{" "}
              {/* 显示错误信息 */}
              <DialogHeader>
                <DialogTitle>添加大模型</DialogTitle>
                <DialogDescription>
                  填写模型配置信息后，点击保存。
                </DialogDescription>
              </DialogHeader>
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="model_id" className="text-right">
                    模型ID
                  </Label>
                  <Input
                    id="model_id"
                    placeholder="model_id"
                    onChange={handleInputChange}
                    className="col-span-3"
                  />
                </div>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="base_url" className="text-right">
                  模型来源
                </Label>
                <div className="col-span-3">
                  <input
                    id="base_url"
                    list="base_url_options"
                    placeholder="输入或选择模型base_url"
                    onChange={handleInputChange}
                    className="w-full border border-gray-300 rounded-md p-2"
                  />
                  <datalist id="base_url_options">
                    <option value="https://api.openai.com/v1">OpenAI</option>
                    <option value="https://dashscope.aliyuncs.com/compatible-mode/v1">
                      通义千问
                    </option>
                    {/* 添加更多预设选项 */}
                  </datalist>
                </div>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="api_key" className="text-right">
                  API Key
                </Label>
                <Input
                  id="api_key"
                  placeholder="api_key"
                  onChange={handleInputChange}
                  className="col-span-3"
                />
              </div>
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="model" className="text-right">
                    模型名称
                  </Label>
                  <Input
                    id="model"
                    placeholder="model"
                    onChange={handleInputChange}
                    className="col-span-3"
                  />
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
