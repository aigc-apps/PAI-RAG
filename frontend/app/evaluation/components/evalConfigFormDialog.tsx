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
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Switch } from '@/components/ui/switch';
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Loader2, ChevronDownIcon } from "lucide-react";
import { useState, useEffect } from 'react';
import { McpConfig } from '@/app/config/mcp/mcp';
import { LlmConfig } from '@/app/config/model/llm/page';
import { KbConfig } from '@/app/knowledgebases/kbconfig';
import { useRouter } from 'next/navigation';
import { EvalRunConfig } from '@/app/evaluation/[evalId]/types';


interface EvalConfigFormDialogProps {
  mode: 'new' | 'edit';
  config?: EvalRunConfig; // edit 时传入
  llms: LlmConfig[];
  mcps: McpConfig[];
  kbs: KbConfig[];
  evalId: string;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (config: EvalRunConfig) => void;
  isSaving: boolean;
}

export function EvalConfigFormDialog({
  mode,
  config,
  llms,
  mcps,
  kbs,
  evalId,
  isOpen,
  onOpenChange,
  onSave,
  isSaving,
}: EvalConfigFormDialogProps) {
  const router = useRouter();
  const [localConfig, setLocalConfig] = useState<EvalRunConfig>(
    mode === 'edit' && config
      ? { ...config }
      : {
        id: "",
        name: "",
        model_id: "",
        mcp_ids: [],
        kb_ids: [],
        enable_search: false,
        enable_vision: false,
        enable_agent: false,
        enable_input_guardrail: false,
        enable_output_guardrail: false,
        guardrail_hint: "作为人工智能助手，我无法回应包含不当或敏感信息的内容。",
        evaluator_config: {
          name: "",
          model_id: "",
          case_sensitive: false,
          ignore_punctuation: false,
        }
      }
  );

  const [selectedKbNames, setSelectedKbNames] = useState<string[]>(
    kbs.filter(kb => localConfig.kb_ids.includes(kb.id)).map(kb => kb.name)
  );
  const [selectedMcpNames, setSelectedMcpNames] = useState<string[]>(
    mcps.filter(mcp => localConfig.mcp_ids.includes(mcp.id)).map(mcp => mcp.name)
  );

  // 当 config 或 mode 变化时重置表单
  useEffect(() => {
    if (mode === 'edit' && config) {
      setLocalConfig({ ...config });
      setSelectedKbNames(
        kbs.filter(kb => config.kb_ids.includes(kb.id)).map(kb => kb.name)
      );
      setSelectedMcpNames(
        mcps.filter(mcp => config.mcp_ids.includes(mcp.id)).map(mcp => mcp.name)
      );
    } else {
      setLocalConfig({
        id: "",
        name: "",
        model_id: "",
        mcp_ids: [],
        kb_ids: [],
        enable_search: false,
        enable_vision: false,
        enable_agent: false,
        enable_input_guardrail: false,
        enable_output_guardrail: false,
        guardrail_hint: "作为人工智能助手，我无法回应包含不当或敏感信息的内容。",
        evaluator_config: {
          name: "",
          model_id: "",
          case_sensitive: false,
          ignore_punctuation: false,
        }
      });
      setSelectedKbNames([]);
      setSelectedMcpNames([]);
    }
  }, [mode, config, kbs, mcps]);

  const handleKbSelect = (kb_id: string, kb_name: string, checked: boolean) => {
    setLocalConfig(prev => {
      const kb_ids = checked
        ? prev.kb_ids.includes(kb_id) ? prev.kb_ids : [...prev.kb_ids, kb_id]
        : prev.kb_ids.filter(id => id !== kb_id);
      return { ...prev, kb_ids };
    });

    setSelectedKbNames(prev => {
      if (checked) {
        return prev.includes(kb_name) ? prev : [...prev, kb_name];
      } else {
        return prev.filter(name => name !== kb_name);
      }
    });
  };

  const handleMcpSelect = (mcp_id: string, mcp_name: string, checked: boolean) => {
    setLocalConfig(prev => {
      const mcp_ids = checked
        ? prev.mcp_ids.includes(mcp_id) ? prev.mcp_ids : [...prev.mcp_ids, mcp_id]
        : prev.mcp_ids.filter(id => id !== mcp_id);
      return { ...prev, mcp_ids };
    });

    setSelectedMcpNames(prev => {
      if (checked) {
        return prev.includes(mcp_name) ? prev : [...prev, mcp_name];
      } else {
        return prev.filter(name => name !== mcp_name);
      }
    });
  };

  const handleSubmit = () => {
    onSave(localConfig);
  };

  const mode_str = mode === "new" ? "新建" : "修改";

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{mode_str}评估配置</DialogTitle>
          <DialogDescription>{mode_str}一个评估配置后，点击保存。</DialogDescription>
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

          {/* 基模型选择 */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label htmlFor="basemodel" className="flex items-center">
              基模型选择 <span className="text-destructive ml-1">*</span>
            </Label>
            <div>
              {llms.length > 0 ? (
                <Select
                  value={localConfig.model_id}
                  onValueChange={(value) =>
                    setLocalConfig((prev) => ({
                      ...prev,
                      model_id: value,
                    }))
                  }
                >
                  <SelectTrigger id="basemodel">
                    <SelectValue placeholder="请选择基模型" />
                  </SelectTrigger>
                  <SelectContent>
                    {llms.map((llm) => (
                      <SelectItem key={llm.id} value={llm.model_id}>
                        {llm.model_id}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ) : (
                <div className="flex flex-col gap-2">
                  <p className="text-sm text-muted-foreground">尚未配置大模型</p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => router.push('/config/model/llm')}
                  >
                    前往添加
                  </Button>
                </div>
              )}
            </div>
          </div>

          {/* 启用联网搜索 */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label htmlFor="enable_search">启用联网搜索</Label>
            <Switch
              id="enable_search"
              className="justify-self-start"
              checked={localConfig.enable_search}
              onCheckedChange={(checked) => {
                setLocalConfig((prev) => ({
                  ...prev,
                  enable_search: checked,
                }));
              }}
            />
          </div>

          {/* Agentic模式 */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label htmlFor="enable_agent">Agentic模式</Label>
            <Switch
              id="enable_agent"
              className="justify-self-start"
              checked={localConfig.enable_agent}
              onCheckedChange={(checked) => {
                setLocalConfig((prev) => ({
                  ...prev,
                  enable_agent: checked,
                }));
              }}
            />
          </div>

          {/* 知识库选择 */}
          <div className="grid grid-cols-[120px_1fr] items-start gap-4">
            <Label htmlFor="kb_selection">知识库选择</Label>
            <div className="space-y-2">
              {kbs.length > 0 ? (
                <DropdownMenu modal={true}>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      className="text-sm text-muted-foreground"
                    >
                      已选{localConfig?.kb_ids.length || 0}个，可多选 <ChevronDownIcon />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent className="w-56">
                    <DropdownMenuLabel>知识库</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    {kbs.map((kb) => (
                      <DropdownMenuCheckboxItem
                        key={kb.id}
                        checked={localConfig.kb_ids.includes(kb.id)}
                        onCheckedChange={(checked) =>
                          handleKbSelect(kb.id, kb.name, checked)
                        }
                        onSelect={(e) => e.preventDefault()}
                      >
                        {kb.name}
                      </DropdownMenuCheckboxItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              ) : (
                <p className="text-sm text-muted-foreground">尚未配置知识库</p>
              )}

              {selectedKbNames.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {selectedKbNames.map((name) => (
                    <Badge variant="secondary" key={name}>
                      {name}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* MCP选择 */}
          <div className="grid grid-cols-[120px_1fr] items-start gap-4">
            <Label htmlFor="mcp_selection">MCP选择</Label>
            <div className="space-y-2">
              {mcps.length > 0 ? (
                <DropdownMenu modal={true}>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      className="text-sm text-muted-foreground"
                    >
                      已选{localConfig.mcp_ids.length}个，可多选 <ChevronDownIcon />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent className="w-56">
                    <DropdownMenuLabel>MCP</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    {mcps.map((mcp) => (
                      <DropdownMenuCheckboxItem
                        key={mcp.id}
                        checked={localConfig.mcp_ids.includes(mcp.id)}
                        onCheckedChange={(checked) =>
                          handleMcpSelect(mcp.id, mcp.name, checked)
                        }
                        onSelect={(e) => e.preventDefault()}
                      >
                        {mcp.name}
                      </DropdownMenuCheckboxItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              ) : (
                <p className="text-sm text-muted-foreground">尚未配置MCP</p>
              )}

              {selectedMcpNames.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {selectedMcpNames.map((name) => (
                    <Badge variant="secondary" key={name}>
                      {name}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* AI安全护栏 */}
          <div className="grid grid-cols-[120px_1fr] items-start gap-4">
            <Label>AI安全护栏</Label>
            <div className="space-y-4">
              {/* 输入/输出护栏 */}
              <div className="grid grid-cols-2 gap-6">
                <div className="flex items-center gap-2">
                  <Switch
                    id="enable_input_check"
                    checked={localConfig.enable_input_guardrail || false}
                    onCheckedChange={(checked) => {
                      setLocalConfig((prev) => ({
                        ...prev,
                        enable_input_guardrail: checked,
                      }));
                    }}
                  />
                  <Label htmlFor="enable_input_check" className="text-sm">
                    输入护栏
                  </Label>
                </div>
                <div className="flex items-center gap-2">
                  <Switch
                    id="enable_output_check"
                    checked={localConfig.enable_output_guardrail || false}
                    onCheckedChange={(checked) => {
                      setLocalConfig((prev) => ({
                        ...prev,
                        enable_output_guardrail: checked,
                      }));
                    }}
                  />
                  <Label htmlFor="enable_output_check" className="text-sm">
                    输出护栏
                  </Label>
                </div>
              </div>

              {/* 默认护栏提示 */}
              <div className="space-y-1">
                <Input
                  id="guardrail_hint"
                  placeholder="作为人工智能助手，我无法回应包含不当或敏感信息的内容。"
                  className="w-full"
                  value={localConfig.guardrail_hint || ""}
                  onChange={(e) => {
                    setLocalConfig((prev) => ({
                      ...prev,
                      guardrail_hint: e.target.value,
                    }));
                  }}
                />
                <Label htmlFor="guardrail_hint" className="text-xs text-muted-foreground">
                  默认护栏提示
                </Label>
              </div>
            </div>
          </div>

          {/* 评估器选择 */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4 border-t  pt-3">
            <Label htmlFor="enable_agent">评估器选择</Label>
            <div className="space-y-4 w-full">
              {/* 评估器类型选择 */}
              <Select
                value={localConfig.evaluator_config?.name || ""}
                onValueChange={(value) => {
                  setLocalConfig((prev) => ({
                    ...prev,
                    evaluator_config: {
                      ...prev.evaluator_config,
                      name: value,
                    },
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
              {localConfig.evaluator_config?.name === "ExactMatch" && (
                <div className="space-y-3 pt-3">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="case_sensitive" className="text-sm">
                      区分大小写
                    </Label>
                    <Switch
                      id="case_sensitive"
                      checked={localConfig.evaluator_config.case_sensitive || false}
                      onCheckedChange={(checked) => {
                        setLocalConfig((prev) => ({
                          ...prev,
                          evaluator_config: {
                            ...prev.evaluator_config!,
                            case_sensitive: checked,
                          },
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
                      checked={localConfig.evaluator_config.ignore_punctuation ?? true}
                      onCheckedChange={(checked) => {
                        setLocalConfig((prev) => ({
                          ...prev,
                          evaluator_config: {
                            ...prev.evaluator_config!,
                            ignore_punctuation: checked,
                          },
                        }));
                      }}
                    />
                  </div>
                </div>
              )}

              {localConfig.evaluator_config?.name === "LLMJudge" && (
                <div className="pt-3">
                  <Label htmlFor="model_id" className="block text-sm mb-2">
                    选择评估器模型
                  </Label>
                  <Select
                    value={localConfig.evaluator_config.model_id || ""}
                    onValueChange={(value) => {
                      setLocalConfig((prev) => ({
                        ...prev,
                        evaluator_config: {
                          ...prev.evaluator_config!,
                          model_id: value,
                        },
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