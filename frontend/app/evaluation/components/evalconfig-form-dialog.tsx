// components/EvalConfigFormDialog.tsx
'use client';

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Loader2, ChevronDownIcon } from "lucide-react";
import { useState, useEffect } from 'react';
import { LlmConfig } from '@/app/config/model/llm/page';
import { useRouter } from 'next/navigation';
import { EvaluatorConfig } from '@/app/evaluation/[datasetId]/types';
import { ResettableTextarea } from '@/app/apps/resetable_textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";


interface EvalConfigFormDialogProps {
  mode: 'new' | 'edit';
  config?: EvaluatorConfig; // edit 时传入
  llms: LlmConfig[];
  datasetId: string;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (config: EvaluatorConfig) => void;
  isSaving: boolean;
}

export function EvalConfigFormDialog({
  mode,
  config,
  llms,
  datasetId,
  isOpen,
  onOpenChange,
  onSave,
  isSaving,
}: EvalConfigFormDialogProps) {
  const router = useRouter();
  const [localConfig, setLocalConfig] = useState<EvaluatorConfig>(
    mode === 'edit' && config
      ? { ...config }
      : {
        id: "",
        name: "",
        type: "",
        model_id: "",
        case_sensitive: false,
        ignore_punctuation: false
      }
  );

  const [llmJudgePrompt, setLlmJudgePrompt] = useState("");
  const [openPrompt, setOpenPrompt] = useState(false);
  const [defaultPrompt, setDefaultPrompt] = useState("");

  // 从 API 加载默认提示词
  useEffect(() => {
    const loadDefaultPrompt = async () => {
      try {
        const response = await fetch('/api/eval-prompts');
        if (response.ok) {
          const data = await response.json();
          const prompt = data.data?.llm_judge_prompt || "";
          setDefaultPrompt(prompt);
        }
      } catch (error) {
        console.error('Failed to load default eval prompt:', error);
      }
    };
    
    loadDefaultPrompt();
  }, []); // 只在组件挂载时执行一次

  // 当 config 或 mode 变化时重置表单
  useEffect(() => {
    if (mode === 'edit' && config) {
      setLocalConfig({ ...config });
      // 安全地读取提示词，如果不存在则使用默认值
      setLlmJudgePrompt(config.llm_judge_prompt || defaultPrompt || "");
    } else {
      setLocalConfig({
        id: "",
        name: "",
        type: "",
        model_id: "",
        case_sensitive: false,
        ignore_punctuation: false
      });
      // 初始化提示词为默认值（使用从 API 加载的默认值）
      setLlmJudgePrompt(defaultPrompt || "");
    }
  }, [mode, config, defaultPrompt]);
  const handleSubmit = () => {
    // 如果类型是 LLMJudge，确保包含 llm_judge_prompt
    // 如果提示词为空字符串，传递 null，这样后端会使用默认值
    const configToSave = {
      ...localConfig,
      ...(localConfig.type === "LLMJudge" && { 
        llm_judge_prompt: llmJudgePrompt.trim() || null 
      }),
    };
    onSave(configToSave);
  };

  const mode_str = mode === "new" ? "新建" : "修改";

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{mode_str}评估器配置</DialogTitle>
          <DialogDescription>{mode_str}一个评估器配置后，点击保存。</DialogDescription>
        </DialogHeader>

        <div className="grid gap-6 py-4">
          {/* 名称 */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label htmlFor="setting_name">配置名称</Label>
            <Input
              id="setting_name"
              value={localConfig.name}
              onChange={(e) =>
                setLocalConfig((prev) => ({ ...prev, name: e.target.value }))
              }
              placeholder="请输入配置名称, 如config_v1"
              required
            />
          </div>

          

          {/* 评估器选择 */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4 border-t  pt-3">
            <Label htmlFor="enable_agent">评估器选择</Label>
            <div className="space-y-4 w-full">
              {/* 评估器类型选择 */}
              <Select
                value={localConfig.type || ""}
                onValueChange={(value) => {
                  setLocalConfig((prev) => ({
                    ...prev,
                    type: value,
                  }));
                }}
              >
                <SelectTrigger id="evaluator_name">
                  <SelectValue placeholder="选择评估器" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="ExactMatch">精确匹配</SelectItem>
                  <SelectItem value="LLMJudge">LLM 评判</SelectItem>
                </SelectContent>
              </Select>

              {/* 动态配置区域 */}
              {localConfig.type === "ExactMatch" && (
                <div className="space-y-3 pt-3">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="case_sensitive" className="text-sm">
                      区分大小写
                    </Label>
                    <Switch
                      id="case_sensitive"
                      checked={localConfig.case_sensitive || false}
                      onCheckedChange={(checked) => {
                        setLocalConfig((prev) => ({
                          ...prev,
                          case_sensitive: checked,
                        }));
                      }}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label htmlFor="ignore_punctuation" className="text-sm">
                      忽略标点符号
                    </Label>
                    <Switch
                      id="ignore_punctuation"
                      checked={localConfig.ignore_punctuation ?? true}
                      onCheckedChange={(checked) => {
                        setLocalConfig((prev) => ({
                          ...prev,
                          ignore_punctuation: checked,
                        }));
                      }}
                    />
                  </div>
                </div>
              )}

              {localConfig.type === "LLMJudge" && (
                <div className="pt-3 space-y-3">
                  <div>
                    <Label htmlFor="model_id" className="block text-sm mb-2">
                      选择评估器模型
                    </Label>
                    <Select
                      value={localConfig.model_id || ""}
                      onValueChange={(value) => {
                        setLocalConfig((prev) => ({
                          ...prev,
                          model_id: value,
                        }));
                      }}
                    >
                      <SelectTrigger id="model_id">
                        <SelectValue placeholder="请选择评估模型" />
                      </SelectTrigger>
                      <SelectContent>
                        {llms.map((llm) => (
                          <SelectItem key={llm.model_id} value={llm.model_id}>
                            {llm.model_id}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  
                  {/* 提示词设置按钮 */}
                  <div className="flex items-center gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => setOpenPrompt(true)}
                    >
                      提示词设置 - 编辑提示词
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* 提示词编辑对话框 */}
        {localConfig.type === "LLMJudge" && (
          <Dialog open={openPrompt} onOpenChange={setOpenPrompt}>
            <DialogContent className="sm:max-w-4xl max-h-[90vh] flex flex-col">
              <DialogHeader>
                <DialogTitle>编辑 LLM 评判提示词</DialogTitle>
                <DialogDescription>
                  编辑 LLM 评判评估器使用的提示词模板。可以使用 {`{inputs}`}、{`{outputs}`}、{`{reference_outputs}`} 作为占位符。
                </DialogDescription>
              </DialogHeader>
              <div className="flex-1 overflow-hidden flex flex-col min-h-0">
                <Tabs defaultValue="prompt" className="flex-1 flex flex-col min-h-0">
                  <TabsList>
                    <TabsTrigger value="prompt">提示词</TabsTrigger>
                  </TabsList>
                  <TabsContent value="prompt" className="flex-1 overflow-hidden flex flex-col min-h-0">
                    <div className="flex-1 overflow-auto">
                      <ResettableTextarea
                        value={llmJudgePrompt}
                        onChange={(e) => setLlmJudgePrompt(e.target.value)}
                        placeholder="请输入 LLM 评判提示词..."
                        className="min-h-[400px] font-mono text-sm"
                        onReset={() => setLlmJudgePrompt(defaultPrompt)}
                      />
                    </div>
                  </TabsContent>
                </Tabs>
              </div>

              <DialogFooter className="gap-2 sm:gap-0">
                <DialogClose asChild>
                  <Button variant="outline" onClick={() => {
                    // 取消时恢复为配置中的值或默认值
                    const prompt = localConfig.llm_judge_prompt || defaultPrompt;
                    setLlmJudgePrompt(prompt);
                  }}>取消</Button>
                </DialogClose>
                <Button type="button" onClick={() => {
                  setOpenPrompt(false);
                }}>
                  保存更改
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        )}

        <DialogFooter className="gap-2 sm:gap-0">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            取消
          </Button>
          <Button onClick={handleSubmit} disabled={isSaving}>
            {isSaving ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                提交中...
              </>
            ) : (
              mode_str
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}