// components/EvalConfigFormDialog.tsx
'use client';

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
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
import { useTracingConfig } from '@/app/config/tracing/traceingconfig'


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
        ignore_punctuation: false,
        extra_params: {},
      }
  );
  const { config: traceConfig, loading: tracingLoading, error: tracingError } = useTracingConfig();

  // 当 config 或 mode 变化时重置表单
  useEffect(() => {
    if (mode === 'edit' && config) {
      setLocalConfig({ ...config });
    } else {
      setLocalConfig({
        id: "",
        name: "",
        type: "",
        model_id: "",
        case_sensitive: false,
        ignore_punctuation: false,
        extra_params: {},
      });
    }
  }, [mode, config]);
  const handleSubmit = () => {
    onSave(localConfig);
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
                  <SelectItem value="AgentTrajectory">LLM 评判(AgentTrajectory)</SelectItem>
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

              {(localConfig.type === "LLMJudge" || localConfig.type === "AgentTrajectory") && (
                <div className="pt-3">
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
                  {(!localConfig.model_id) && (
                    <p style={{ color: 'red', fontSize: '12px' }}>
                      请从列表中选择一个模型
                    </p>
                  )}
                </div>
              )}

              {localConfig.type === "AgentTrajectory" && (
                <div className="pt-3">
                  {localConfig.type === "AgentTrajectory" && (
                    <div className="mt-4 space-y-2 text-sm">
                      {tracingLoading ? (
                        <p className="text-gray-500">正在加载链路追踪配置...</p>
                      ) : tracingError ? (
                        <p className="text-red-500">加载失败: {tracingError}</p>
                      ) : traceConfig?.enabled ? (
                        <>
                           <div className="flex items-center">
                            <span className="w-20 text-green-700 font-medium">链路追踪 </span>
                            <span className="text-green-700 font-medium">已启用</span>
                          </div>
                          <div className="flex items-center">
                            <Label className="w-20 text-gray-600">Region</Label>
                            <span>{traceConfig.region || '未知区域'}</span>
                          </div>
                          <div className="flex items-center">
                            <Label className="w-20 text-gray-600">服务名</Label>
                            <span>{traceConfig.service_name}</span>
                          </div>
                          <div className="flex items-center">
                            <Label className="w-20 text-gray-600">Endpoint</Label>
                            <span>{traceConfig.endpoint}</span>
                          </div>

                          <div className="space-y-2 mt-3 border-t pt-3">
                            <Label className="block">认证信息</Label>
                            <div className="space-y-3">
                              {/* Access Key */}
                              <div className="flex items-center">
                                <Label htmlFor="ak" className="min-w-20">Access Key</Label>
                                <Input
                                  id="access_key_id"
                                  type="password"
                                  value={localConfig.extra_params?.access_key_id || ""}
                                  onChange={(e) => {
                                    setLocalConfig((prev) => ({
                                      ...prev,
                                      extra_params: {
                                        ...prev.extra_params,
                                        access_key_id: e.target.value,
                                        region: traceConfig.region,
                                      },
                                    }));
                                  }}
                                  placeholder="请输入 Access Key ID"
                                  className="flex-1"
                                />
                              </div>

                              <div className="flex items-center">
                                <Label htmlFor="access_key_secret" className="min-w-20">Secret Key</Label>
                                <Input
                                  id="access_key_secret"
                                  type="password"
                                  value={localConfig.extra_params?.access_key_secret || ""}
                                  onChange={(e) => {
                                    setLocalConfig((prev) => ({
                                      ...prev,
                                      extra_params: {
                                        ...prev.extra_params,
                                        access_key_secret: e.target.value,
                                        region: traceConfig.region,
                                      },
                                    }));
                                  }}
                                  placeholder="请输入 Access Key Secret"
                                  className="flex-1"
                                />
                              </div>
                            </div>
                          </div>
                        </>
                      ) : (
                        <p className="text-yellow-600">⚠️ 当前未启用链路追踪，请先配置并启用链路追踪！</p>
                      )}
                    </div>
                  )}
                </div>
              )}

            </div>
          </div>
        </div>

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