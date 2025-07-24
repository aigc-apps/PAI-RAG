// RerankerModelDialog.tsx
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
import { Alert, AlertTitle } from "@/components/ui/alert";
import { AlertCircleIcon } from "lucide-react";

// 定义组件 props
interface RerankerModelDialogProps {
  isAdd: boolean;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  rerankerConfig: RerankerConfig;
  onSaveSuccess: (reranker: RerankerConfig) => void;
}

interface RerankerConfig {
  id: string;
  model_id: string;
  model_name: string;
  api_key: string;
  base_url: string;
}

export const RerankerModelDialog: FC<RerankerModelDialogProps> = ({
  isAdd,
  isOpen,
  setIsOpen,
  rerankerConfig,
  onSaveSuccess,
}) => {
  const [reranker, setReranker] = useState<RerankerConfig>(rerankerConfig);
  const [error, setError] = useState<string | null>(null);
  const [saveErrorMsg, setSaveErrorMsg] = useState(""); // 保存错误信息

  useEffect(() => {
    setReranker(rerankerConfig);
  }, [isAdd, rerankerConfig]);

  useEffect(() => {
    setSaveErrorMsg("");
  }, [reranker]);

  const handleSubmit = async () => {
    setSaveErrorMsg("");
    console.log("reranker", reranker);
    if (
      !reranker.model_id ||
      (isAdd && !reranker.api_key) ||
      !reranker.model_name ||
      !reranker.base_url
    ) {
      setSaveErrorMsg("请必须填写完整的模型信息");
      return;
    }
    const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
    const submit_url = isAdd
      ? `http://localhost:${port}/v1/config/rerankers`
      : `http://localhost:${port}/v1/config/rerankers/${reranker.id}`;
    const updateMethod = isAdd ? "POST" : "PATCH";
    if (reranker.api_key === "******") reranker.api_key = "";
    console.log("updateMethod", isAdd, updateMethod, submit_url, reranker);
    try {
      const res = await fetch(submit_url, {
        method: updateMethod,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reranker),
      });

      if (!res.ok) {
        setSaveErrorMsg(`${updateMethod} 请求失败, 请检查填写信息`);
        return;
      }
      const jsondata = await res.json();
      onSaveSuccess(jsondata.data as RerankerConfig); // 触发回调
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
              value={reranker?.model_id ?? ""}
              onChange={(e) =>
                setReranker((prev) => ({ ...prev, model_id: e.target.value }))
              }
              className="col-span-3"
            />
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="model_name" className="text-right">
              模型名称
              <span className="text-destructive">*</span>
            </Label>
            <Input
              id="model_name"
              placeholder="model_name"
              value={reranker?.model_name ?? ""}
              onChange={(e) =>
                setReranker((prev) => ({ ...prev, model_name: e.target.value }))
              }
              className="col-span-3"
            />
          </div>
        </div>
        <div>
          <div className="grid grid-cols-4 items-center gap-4 py-2">
            <Label htmlFor="base_url" className="text-right">
              Base URL
              <span className="text-destructive">*</span>
            </Label>
            <div className="col-span-3">
              <input
                id="base_url"
                list="base_url_options"
                placeholder="输入或选择模型base_url"
                value={reranker?.base_url ?? ""}
                onChange={(e) =>
                  setReranker((prev) => ({ ...prev, base_url: e.target.value }))
                }
                className="w-full border border-gray-300 rounded-md p-2 text-sm"
              />
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
                value={reranker?.api_key ?? ""}
                onChange={(e) =>
                  setReranker((prev) => ({ ...prev, api_key: e.target.value }))
                }
                className="col-span-3"
              />
            ) : (
              <Input
                id="api_key"
                type="password"
                placeholder="api_key"
                value={reranker?.api_key || "******"}
                onChange={(e) =>
                  setReranker((prev) => ({ ...prev, api_key: e.target.value }))
                }
                className="col-span-3"
              />
            )}
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
