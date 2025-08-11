// LLMModelDialog.tsx
import { useState, useEffect, FC } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Alert, AlertTitle } from "@/components/ui/alert";
import { AlertCircleIcon } from "lucide-react";

// 定义组件 props
interface LLMModelDialogProps {
  isAdd: boolean;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  llmConfig: LlmConfig;
  onSaveSuccess: (llm: LlmConfig) => void;
}

// 模型数据类型
interface LlmConfig {
  id: string;
  model_id: string;
  source: string;
  model: string;
  api_key: string;
  base_url: string;
  max_context: number;
  enabled: boolean;
  vision_support: boolean;
}

export const LLMModelDialog: FC<LLMModelDialogProps> = ({
  isAdd,
  isOpen,
  setIsOpen,
  llmConfig,
  onSaveSuccess,
}) => {
  const [llm, setLlm] = useState<LlmConfig>(llmConfig);
  const [error, setError] = useState<string | null>(null);
  const [saveErrorMsg, setSaveErrorMsg] = useState(""); // 保存错误信息

  useEffect(() => {
    setLlm(llmConfig);
  }, [isAdd, llmConfig]);

  useEffect(() => {
    setSaveErrorMsg("");
  }, [llm]);

  const handleSubmit = async () => {
    setSaveErrorMsg("");
    if (
      !llm.model ||
      (isAdd && !llm.api_key) ||
      !llm.base_url ||
      !llm.model_id
    ) {
      setSaveErrorMsg("请必须填写完整的模型信息");
      return;
    }
    const submit_url = isAdd ? "/v1/config/llms" : `/v1/config/llms/${llm.id}`;
    const updateMethod = isAdd ? "POST" : "PATCH";
    if (llm.api_key === "******") llm.api_key = "";
    console.log("updateMethod", isAdd, updateMethod, submit_url, llm);
    try {
      const res = await fetch(submit_url, {
        method: updateMethod,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(llm),
      });

      if (!res.ok) {
        setSaveErrorMsg(`${updateMethod} 请求失败, 请检查填写信息`);
        return;
      }
      const jsondata = await res.json();
      onSaveSuccess(jsondata.data as LlmConfig); // 触发回调
      setIsOpen(false);
    } catch (err: any) {
      setSaveErrorMsg(`${updateMethod} 请求失败`);
    } finally {
    }
  };

  // 对话框关闭时重置表单
  const handleDialogClose = (open: boolean) => {
    setIsOpen(open);
    if (!open) {
      setError(null);
      setSaveErrorMsg("");
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleDialogClose}>
      <DialogContent className="sm:max-w-[700px]">
        {error && <div className="text-red-500 mb-4">{error}</div>}
        <DialogHeader>
          <DialogTitle>{isAdd ? "添加模型" : "编辑模型"}</DialogTitle>
          <DialogDescription>填写模型配置信息后，点击保存。</DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 py-2">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="model_id" className="text-right">
              模型ID
              <span className="text-destructive">*</span>
            </Label>
            <Input
              id="model_id"
              placeholder="model_id"
              value={llm?.model_id ?? ""}
              onChange={(e) =>
                setLlm((prev) => ({ ...prev, model_id: e.target.value }))
              }
              className="col-span-3"
            />
          </div>
        </div>
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
                value={llm?.base_url ?? ""}
                onChange={(e) =>
                  setLlm((prev) => ({ ...prev, base_url: e.target.value }))
                }
                className="w-full border border-gray-300 rounded-md p-2 text-sm"
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
          <div className="grid grid-cols-4 items-center gap-4 py-2">
            <Label htmlFor="api_key" className="text-right">
              API Key
              <span className="text-destructive">*</span>
            </Label>
            {isAdd ? (
              <Input
                id="api_key"
                type="password"
                placeholder="api_key"
                value={llm?.api_key ?? ""}
                onChange={(e) =>
                  setLlm((prev) => ({ ...prev, api_key: e.target.value }))
                }
                className="col-span-3"
              />
            ) : (
              <Input
                id="api_key"
                type="password"
                placeholder="api_key"
                value={llm?.api_key || "******"}
                onChange={(e) =>
                  setLlm((prev) => ({ ...prev, api_key: e.target.value }))
                }
                className="col-span-3"
              />
            )}
          </div>
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
              value={llm?.model ?? ""}
              onChange={(e) =>
                setLlm((prev) => ({ ...prev, model: e.target.value }))
              }
              className="col-span-3"
            />
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="vision_support" className="text-right">
              多模态模型
            </Label>
            <Switch
              id="vision_support"
              checked={llm?.vision_support ?? false}
              onCheckedChange={(checked) =>
                setLlm((prev) => ({ ...prev, vision_support: checked }))
              }
            />
          </div>
        </div>

        <DialogFooter className="flex flex-col gap-4">
          {saveErrorMsg !== "" && (
            <Alert className="bg-destructive/10 dark:bg-destructive/20 border-none">
              <AlertCircleIcon className="h-4 w-4 !text-destructive" />
              <AlertTitle>{saveErrorMsg}</AlertTitle>
            </Alert>
          )}
          <Button onClick={handleSubmit}>{isAdd ? "新增" : "保存"}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
