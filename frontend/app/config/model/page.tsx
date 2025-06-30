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
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";

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

interface LlmConfig {
  id: string;
  model_id: string;
  source: string;
  model: string;
  api_key: string;
  base_url: string;
  max_context: number;
  enabled: boolean;
}

interface EmbConfig {
  id: string;
  model_id: string;
  model_name: string;
  type: string;
  api_key: string;
  endpoint: string;
  dimension: number;
  embed_batch_size: number;
}

// 联合类型
type ModelConfig = LlmConfig | EmbConfig;

const isLlmConfig = (config: ModelConfig | null): config is LlmConfig => {
  return config !== null && "base_url" in config;
};

const isEmbConfig = (config: ModelConfig | null): config is EmbConfig => {
  return config !== null && "endpoint" in config;
};

export default function ModelConfigPage() {
  const [isOpen, setIsOpen] = useState(false); // 控制 AddLlmDialog 显示
  const [isEditOpen, setIsEditOpen] = useState(false); // 控制 EditLlmDialog 显示
  const [editingConfig, setEditingConfig] = useState<ModelConfig | null>(null); // 当前编辑的配置
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
    source: "",
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key: "sk-xxxxxx",
    enabled: true,
    dimension: 1024, // 默认向量维度
    embed_batch_size: 10, // 默认向量Batch大小
  });

  const [modelconfigs, setModelConfigs] = useState<ModelConfig[]>([]); // 存储 LLM 配置
  const [modelloading, setModelLoading] = useState(true); // 加载状态
  const [modelerror, setModelError] = useState(""); // 错误信息

  const [selectedModelType, setSelectedModelType] = useState("llm"); // 默认值为 "llm"
  const [selectedEmbeddingModelType, setSelectedEmbeddingModelType] =
    useState("local"); // 默认值为 "local"

  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
        const [llmRes, embRes] = await Promise.all([
          fetch(`http://localhost:${port}/v1/config/llms`),
          fetch(`http://localhost:${port}/v1/config/embeddings`),
        ]);

        const llmData = await llmRes.json();
        const embData = (await embRes.json())?.data || [];
        console.log("llmData", llmData);
        console.log("embData", embData);
        setModelConfigs([...llmData, ...embData]); // 合并
      } catch (err: any) {
        setModelError(err || "加载失败");
      } finally {
        setModelLoading(false);
      }
    };
    fetchModelConfigs();
  }, []);

  const handleEditClick = (config: ModelConfig) => {
    console.log("handleEditClick", config);
    setEditingConfig({ ...config }); // 深拷贝当前配置
    setIsEditOpen(true);
  };
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    setAddFormData((prev) => ({ ...prev, [id]: value }));
  };
  const addModel = async () => {
    try {
      const newModel = {
        ...addFormData,
      };
      console.log("newModel:", newModel);
      console.log("selectModelType:", selectedModelType);
      console.log("selectEmbeddingModelType:", selectedEmbeddingModelType);
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;

      if (selectedModelType === "llm") {
        const newLlmModel: LlmConfig = {
          id: uuidv4(),
          base_url: newModel.base_url,
          model: newModel.model,
          model_id: newModel.model_id,
          enabled: newModel.enabled,
          api_key: newModel.api_key,
          source: newModel.source,
          max_context: 0,
        };
        console.log("newLlmModel:", newLlmModel);
        const res = await fetch(`http://localhost:${port}/v1/config/llms`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(newLlmModel), // 包装为数组
        });
        if (!res.ok) throw new Error("添加 LLM 模型配置失败");
        setModelConfigs((prev) => [...prev, newLlmModel]); // 追加新 LLM 配置
      } else {
        const newEmbModel: EmbConfig = {
          id: uuidv4(),
          model_name: newModel.model,
          dimension: newModel.dimension,
          endpoint: newModel.base_url,
          type: selectedEmbeddingModelType,
          api_key: newModel.api_key,
          embed_batch_size: newModel.embed_batch_size,
          model_id: newModel.model_id,
        };
        console.log("newEmbModel:", newEmbModel);
        const res = await fetch(
          `http://localhost:${port}/v1/config/embeddings`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(newEmbModel), // 包装为数组
          },
        );
        if (!res.ok) throw new Error("添加 Embedding 模型配置失败");
        setModelConfigs((prev) => [...prev, newEmbModel]); // 追加新 Embedding 配置
      }

      setToastState({
        open: true,
        title: "配置已添加",
        description: "新模型配置已成功保存",
        variant: "default",
      });
      setIsOpen(false); // 关闭 AddDialog
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

  const updateModel = async (model_type: string) => {
    try {
      if (!editingConfig) return;

      if (editingConfig.api_key === "******") editingConfig.api_key = "";
      console.log("editingConfig", editingConfig);

      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
      const res = await fetch(
        `http://localhost:${port}/v1/config/${model_type}/${editingConfig.id}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(editingConfig), // 包装为数组
        },
      );

      if (!res.ok) throw new Error(`修改 ${model_type} 配置失败`);
      setToastState({
        open: true,
        title: `${model_type} 配置已修改`,
        description: "修改的模型配置已成功保存",
        variant: "default",
      });
      setIsEditOpen(false); // 关闭 EditDialog
      console.log(`update ${model_type}`, editingConfig);
      setModelConfigs((prev) =>
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

  const removeModel = async (id: string, model_type: string) => {
    try {
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
      const res = await fetch(
        `http://localhost:${port}/v1/config/${model_type}/${id}`,
        {
          method: "DELETE",
          headers: {
            "Content-Type": "application/json",
          },
        },
      );

      if (!res.ok) {
        throw new Error(`${model_type}删除失败，请检查网络或配置`);
      }

      // 显示成功提示（可选）
      setToastState({
        open: true,
        title: "删除成功",
        description: `${model_type} 配置已移除`,
        variant: "default",
      });

      // 删除成功后更新本地状态
      setModelConfigs((prev) => prev.filter((config) => config.id !== id));
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

  const handleActivateToggle = async (config: LlmConfig) => {
    console.log("handleActivateToggle", config);
    const updatedConfig = {
      ...config,
      enabled: !config.enabled,
    };
    console.log("updatedConfig", updatedConfig);

    try {
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8097;
      const res = await fetch(
        `http://localhost:${port}/v1/config/llms/${updatedConfig.id}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(updatedConfig), // 包装为数组
        },
      );

      if (!res.ok) throw new Error("更新 LLM 状态失败");
      setToastState({
        open: true,
        title: "LLM 状态已更新",
        description: "LLM 状态已更新成功",
        variant: "default",
      });

      setModelConfigs((prev) =>
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
          {modelloading ? (
            <div className="py-12 text-center">
              <p className="text-gray-500">加载中...</p>
            </div>
          ) : modelerror ? (
            <div className="py-12 text-center text-red-500">
              <p>{modelerror}</p>
            </div>
          ) : modelconfigs.length === 0 ? (
            <h3 className="text-lg font-medium text-gray-700 py-6">暂无模型</h3>
          ) : (
            <div className="gap-6 p-4 w-full">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {modelconfigs.map((config) => (
                  <div
                    key={config.id}
                    className="bg-white rounded-lg shadow-sm border border-gray-100 p-4 hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-center gap-2 mb-3">
                      {isLlmConfig(config) ? (
                        <div>
                          <h3 className="font-medium text-gray-800">
                            {config.model}
                          </h3>
                          <span className="text-xs bg-blue-50 text-blue-500 px-2 py-0.5 rounded-full">
                            LLM
                          </span>
                          <span className="text-xs bg-blue-50 text-blue-500 px-2 py-0.5 rounded-full">
                            {config.source}
                          </span>
                        </div>
                      ) : (
                        <div>
                          <h3 className="font-medium text-gray-800">
                            {config.model_name}
                          </h3>
                          <span className="text-xs bg-red-50 text-red-500 px-2 py-0.5 rounded-full">
                            Embedding
                          </span>
                          <span className="text-xs bg-red-50 text-red-500 px-2 py-0.5 rounded-full">
                            {config.type === "local" ? "本地模型" : "API模型"}
                          </span>
                        </div>
                      )}
                    </div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-600">
                        模型ID: {config.model_id}
                      </p>
                      {isLlmConfig(config) ? (
                        <p className="text-sm font-mono px-2 py-4 rounded text-gray-800 truncate">
                          API地址: {config.base_url}
                        </p>
                      ) : (
                        <p className="text-sm font-mono px-2 py-4 rounded text-gray-800 truncate">
                          API地址: {config.endpoint}
                        </p>
                      )}
                    </div>
                    <div className="flex justify-end gap-2 pt-2 border-t border-gray-100">
                      {isLlmConfig(config) && (
                        <Switch
                          checked={config.enabled}
                          onCheckedChange={() => handleActivateToggle(config)}
                          className="ml-auto"
                        />
                      )}
                      <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
                        <DialogTrigger asChild>
                          <button
                            className="text-black-100 hover:text-black-100 px-1 py-1"
                            onClick={() => handleEditClick(config)}
                          >
                            <Edit className="w-4 h-4" />
                          </button>
                        </DialogTrigger>
                        <DialogContent className="sm:max-w-[700px]">
                          {error && (
                            <div className="text-red-500 mb-4">{error}</div>
                          )}{" "}
                          <DialogHeader>
                            <DialogTitle>编辑模型配置</DialogTitle>
                            <DialogDescription>
                              编辑模型配置信息后，点击保存。{editingConfig?.id}
                            </DialogDescription>
                          </DialogHeader>
                          <div className="grid gap-4 py-4">
                            <div className="grid grid-cols-4 items-center gap-4">
                              <Label className="text-right">模型ID</Label>
                              <Input
                                defaultValue={editingConfig?.model_id || "null"}
                                disabled
                                className="col-span-3"
                              />
                            </div>
                            <div className="grid grid-cols-4 items-center gap-4">
                              <Label
                                htmlFor="edit_model"
                                className="text-right"
                              >
                                模型名称
                              </Label>
                              {isLlmConfig(editingConfig) ? (
                                <Input
                                  defaultValue={editingConfig?.model || "null"}
                                  onChange={(e) =>
                                    setEditingConfig({
                                      ...editingConfig,
                                      model: e.target.value,
                                    })
                                  }
                                  className="col-span-3"
                                />
                              ) : (
                                <Input
                                  defaultValue={
                                    editingConfig?.model_name || "null"
                                  }
                                  onChange={(e) =>
                                    setEditingConfig(
                                      editingConfig
                                        ? {
                                            ...editingConfig,
                                            id: editingConfig.id ?? "",
                                            model_name: e.target.value,
                                          }
                                        : null,
                                    )
                                  }
                                  className="col-span-3"
                                />
                              )}
                            </div>
                            <div>
                              {isLlmConfig(editingConfig) ? (
                                <div className="grid grid-cols-4 items-center gap-4 py-2">
                                  <Label className="text-right">
                                    Endpoint URL
                                    <span className="text-destructive">*</span>
                                  </Label>
                                  <div className="col-span-3">
                                    <input
                                      list="base_url_options"
                                      placeholder="输入或选择模型base_url"
                                      defaultValue={editingConfig.base_url}
                                      onChange={(e) =>
                                        setEditingConfig({
                                          ...editingConfig,
                                          base_url: e.target.value,
                                        })
                                      }
                                      className="w-full border border-gray-300 rounded-md p-2 text-sm"
                                    />
                                    <datalist id="base_url_options">
                                      <option value="https://api.openai.com/v1">
                                        OpenAI
                                      </option>
                                      <option value="https://dashscope.aliyuncs.com/compatible-mode/v1">
                                        通义千问
                                      </option>
                                      {/* 添加更多预设选项 */}
                                    </datalist>
                                  </div>
                                </div>
                              ) : (
                                <div className="grid grid-cols-4 items-center gap-4">
                                  <Label
                                    htmlFor="model_type"
                                    className="text-right"
                                  >
                                    模型类型
                                    <span className="text-destructive">*</span>
                                  </Label>
                                  <RadioGroup
                                    className="flex flex-row gap-6 col-span-3"
                                    value={editingConfig?.type}
                                    onValueChange={(value) =>
                                      setEditingConfig(
                                        editingConfig
                                          ? {
                                              ...editingConfig,
                                              id: editingConfig.id ?? "",
                                              type: value,
                                            }
                                          : null,
                                      )
                                    }
                                  >
                                    <div className="flex items-center gap-3">
                                      <RadioGroupItem value="local" />
                                      <Label>本地 (Local Hosted)</Label>
                                    </div>
                                    <div className="flex items-center gap-3">
                                      <RadioGroupItem value="openai_like" />
                                      <Label>API (OpenAI Like)</Label>
                                    </div>
                                  </RadioGroup>
                                </div>
                              )}
                            </div>
                            <div>
                              {isEmbConfig(editingConfig) &&
                                editingConfig?.type === "openai_like" && (
                                  <div className="grid grid-cols-4 items-center gap-4 py-2">
                                    <Label className="text-right">
                                      Endpoint URL
                                      <span className="text-destructive">
                                        *
                                      </span>
                                    </Label>
                                    <div className="col-span-3">
                                      <input
                                        list="base_url_options"
                                        placeholder="输入或选择模型base_url"
                                        defaultValue={editingConfig.endpoint}
                                        onChange={(e) =>
                                          setEditingConfig({
                                            ...editingConfig,
                                            endpoint: e.target.value,
                                          })
                                        }
                                        className="w-full border border-gray-300 rounded-md p-2 text-sm"
                                      />
                                      <datalist id="base_url_options">
                                        <option value="https://api.openai.com/v1">
                                          OpenAI
                                        </option>
                                        <option value="https://dashscope.aliyuncs.com/compatible-mode/v1">
                                          通义千问
                                        </option>
                                        {/* 添加更多预设选项 */}
                                      </datalist>
                                    </div>
                                  </div>
                                )}
                            </div>
                            <div>
                              {(isLlmConfig(editingConfig) ||
                                editingConfig?.type === "openai_like") && (
                                <div className="grid grid-cols-4 items-center gap-4">
                                  <Label
                                    htmlFor="edit_api_key"
                                    className="text-right"
                                  >
                                    API Key
                                  </Label>
                                  <Input
                                    type="password"
                                    defaultValue={
                                      editingConfig?.api_key || "******"
                                    }
                                    onChange={(e) =>
                                      setEditingConfig(
                                        editingConfig
                                          ? {
                                              ...editingConfig,
                                              id: editingConfig.id ?? "",
                                              api_key: e.target.value,
                                            }
                                          : null,
                                      )
                                    }
                                    className="col-span-3"
                                  />
                                </div>
                              )}
                            </div>
                            <div>
                              {isEmbConfig(editingConfig) && (
                                <div>
                                  <div className="grid grid-cols-4 items-center gap-4 py-2">
                                    <Label
                                      htmlFor="dimension"
                                      className="text-right"
                                    >
                                      向量维度
                                    </Label>
                                    <div className="col-span-3">
                                      <Input
                                        id="dimension"
                                        type="number"
                                        placeholder="向量维度"
                                        defaultValue={
                                          editingConfig?.dimension || "null"
                                        }
                                        onChange={(e) =>
                                          setEditingConfig({
                                            ...editingConfig,
                                            dimension: Number(e.target.value),
                                          })
                                        }
                                      />
                                    </div>
                                  </div>
                                  <div className="grid grid-cols-4 items-center gap-4 py-2">
                                    <Label
                                      htmlFor="embed_batch_size"
                                      className="text-right"
                                    >
                                      向量Batch大小
                                    </Label>
                                    <Input
                                      id="embed_batch_size"
                                      type="number"
                                      placeholder="向量Batch大小"
                                      defaultValue={
                                        editingConfig?.embed_batch_size ||
                                        "null"
                                      }
                                      onChange={(e) =>
                                        setEditingConfig({
                                          ...editingConfig,
                                          embed_batch_size: Number(
                                            e.target.value,
                                          ),
                                        })
                                      }
                                      className="col-span-3"
                                    />
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                          <DialogFooter>
                            {isLlmConfig(editingConfig) ? (
                              <Button
                                onClick={() => updateModel("llms")}
                                type="submit"
                                disabled={isEditLoading}
                              >
                                {isEditLoading ? "提交中..." : "修改"}
                              </Button>
                            ) : (
                              <Button
                                onClick={() => updateModel("embeddings")}
                                type="submit"
                                disabled={isEditLoading}
                              >
                                {isEditLoading ? "提交中..." : "修改"}
                              </Button>
                            )}
                          </DialogFooter>
                        </DialogContent>
                      </Dialog>
                      {isLlmConfig(config) ? (
                        <button
                          onClick={() => removeModel(config.id, "llms")}
                          className="text-red-500 hover:text-red-700 p-1"
                          aria-label="删除"
                        >
                          <TrashIcon className="w-4 h-4" />
                        </button>
                      ) : (
                        <button
                          onClick={() => removeModel(config.id, "embeddings")}
                          className="text-red-500 hover:text-red-700 p-1"
                          aria-label="删除"
                        >
                          <TrashIcon className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          <SettingsIcon className="w-10 h-6 text-gray-400 mb-4" />
          <p className="text-gray-500 mt-1">点击下方按钮添加新的模型</p>
          <Dialog open={isOpen} onOpenChange={setIsOpen}>
            <DialogTrigger asChild>
              <Button className="mt-4 px-4 py-2 text-white rounded-lg transition-colors">
                添加模型
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[700px]">
              {error && <div className="text-red-500 mb-4">{error}</div>}{" "}
              {/* 显示错误信息 */}
              <DialogHeader>
                <DialogTitle>添加模型</DialogTitle>
                <DialogDescription>
                  填写模型配置信息后，点击保存。
                </DialogDescription>
              </DialogHeader>
              <div className="grid gap-4 py-2">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="model_type" className="text-right">
                    模型类型
                    <span className="text-destructive">*</span>
                  </Label>
                  <RadioGroup
                    className="col-span-3"
                    value={selectedModelType}
                    onValueChange={setSelectedModelType}
                  >
                    <div className="flex items-center gap-3">
                      <RadioGroupItem value="llm" />
                      <Label>LLM (Large Language Model)</Label>
                    </div>
                    <div className="flex items-center gap-3">
                      <RadioGroupItem value="embedding" />
                      <Label>Embedding (Embedding Model)</Label>
                    </div>
                  </RadioGroup>
                </div>
              </div>
              <div>
                {selectedModelType === "embedding" && (
                  <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="base_url" className="text-right">
                      模型来源
                      <span className="text-destructive">*</span>
                    </Label>
                    <RadioGroup
                      className="flex flex-row gap-6 col-span-3"
                      value={selectedEmbeddingModelType}
                      onValueChange={setSelectedEmbeddingModelType}
                    >
                      <div className="flex items-center gap-3">
                        <RadioGroupItem value="local" />
                        <Label>本地 (Local Hosted)</Label>
                      </div>
                      <div className="flex items-center gap-3">
                        <RadioGroupItem value="openai_like" />
                        <Label>API (OpenAI Like)</Label>
                      </div>
                    </RadioGroup>
                  </div>
                )}
              </div>
              <div className="grid gap-4 py-2">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="model_id" className="text-right">
                    模型ID
                    <span className="text-destructive">*</span>
                  </Label>
                  <Input
                    id="model_id"
                    placeholder="model_id"
                    onChange={handleInputChange}
                    className="col-span-3"
                  />
                </div>
              </div>
              <div>
                {(selectedModelType === "llm" ||
                  selectedEmbeddingModelType === "openai_like") && (
                  <div>
                    <div className="grid grid-cols-4 items-center gap-4 py-2">
                      <Label htmlFor="base_url" className="text-right">
                        Endpoint URL
                        <span className="text-destructive">*</span>
                      </Label>
                      <div className="col-span-3">
                        <input
                          id="base_url"
                          list="base_url_options"
                          placeholder="输入或选择模型base_url"
                          onChange={handleInputChange}
                          className="w-full border border-gray-300 rounded-md p-2 text-sm"
                        />
                        <datalist id="base_url_options">
                          <option value="https://api.openai.com/v1">
                            OpenAI
                          </option>
                          <option value="https://dashscope.aliyuncs.com/compatible-mode/v1">
                            通义千问
                          </option>
                          {/* 添加更多预设选项 */}
                        </datalist>
                      </div>
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4 py-2">
                      <Label htmlFor="api_key" className="text-right">
                        API Key
                        <span className="text-destructive">*</span>
                      </Label>
                      <Input
                        id="api_key"
                        placeholder="api_key"
                        onChange={handleInputChange}
                        className="col-span-3"
                      />
                    </div>
                  </div>
                )}
              </div>
              <div className="grid gap-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="model" className="text-right">
                    模型名称
                    <span className="text-destructive">*</span>
                  </Label>
                  <Input
                    id="model"
                    placeholder="model"
                    onChange={handleInputChange}
                    className="col-span-3"
                  />
                </div>
              </div>
              <div>
                {selectedModelType === "embedding" && (
                  <div>
                    <div className="grid grid-cols-4 items-center gap-4 py-2">
                      <Label htmlFor="dimension" className="text-right">
                        向量维度
                      </Label>
                      <div className="col-span-3">
                        <Input
                          id="dimension"
                          type="number"
                          placeholder="向量维度"
                          onChange={handleInputChange}
                        />
                      </div>
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4 py-2">
                      <Label htmlFor="embed_batch_size" className="text-right">
                        向量Batch大小
                      </Label>
                      <Input
                        id="embed_batch_size"
                        type="number"
                        placeholder="向量Batch大小"
                        onChange={handleInputChange}
                        className="col-span-3"
                      />
                    </div>
                  </div>
                )}
              </div>
              <DialogFooter>
                <Button onClick={addModel} type="submit" disabled={isLoading}>
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
