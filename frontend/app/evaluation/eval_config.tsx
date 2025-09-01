'use client';
import React, { useState, useEffect, FC } from 'react';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { ChevronDownIcon, Terminal } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

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

import { Button } from '@/components/ui/button';
import { McpConfig } from '@/app/config/mcp/mcp';
import { LlmConfig } from '@/app/config/model/llm/page';
import { KbConfig } from '@/app/knowledgebases/kbconfig';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { useRouter } from 'next/navigation';

export interface Chatbot {
  id: string;
  app_id: string;
  description: string;
  enable_search: boolean;
  enable_agent: boolean;
  mcp_ids: string[];
  kb_ids: string[];
  model_id: string;
  updated_at: string;
}

interface ChatbotConfigProps {
  chatbotId: string | undefined;
}

const default_chat_config = {
  id: '',
  app_id: '',
  description: '',
  enable_search: false,
  mcp_ids: [],
  kb_ids: [],
  model_id: "",
  updated_at: "",
  enable_agent: false,
};


interface EvalConfigProps {
  eval_id: string | undefined;
}


// 知识库配置卡片
// export function EvalConfigCard({ eval_id }: { eval_id: string }) {

export const EvalConfigCard: FC<EvalConfigProps> = ({
  eval_id,
}) => {
  const [botConfig, setBotConfig] = useState<Chatbot>(default_chat_config);
  const [llms, setLlms] = useState<LlmConfig[]>([]);
  const [mcps, setMcps] = useState<McpConfig[]>([]);
  const [kbs, setKbs] = useState<KbConfig[]>([]);
  const [selectedKbNames, setSelectedKbNames] = useState<string[]>([]);
  const [selectedMcpNames, setSelectedMcpNames] = useState<string[]>([]);
  const [saveErrorMsg, setSaveErrorMsg] = useState('');
  const isCreate: boolean = eval_id === undefined || eval_id === '';
  const router = useRouter();
  console.log("isCreate", isCreate)
  return (
    <div className="grid gap-4 py-6 px-6">

      <div className="space-y-2">
        <Label htmlFor="app-id">
          实验名称 <span className="text-destructive">*</span>
        </Label>
        <Input
          id="appid"
          value={botConfig.app_id}
          onChange={(e) =>
            setBotConfig((prev) => ({ ...prev, app_id: e.target.value }))
          }
          placeholder="请输入实验名称, 如GAIA"
          required
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="description">描述</Label>
        <Textarea
          id="description"
          value={botConfig.description}
          onChange={(e) =>
            setBotConfig((prev) => ({
              ...prev,
              description: e.target.value,
            }))
          }
          placeholder="评估实验描述（可选）"
          rows={3}
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="description">实验设置</Label>
        <Tabs defaultValue="from-apps" className='space-y-2'>
          <TabsList className="py-4 bg-muted rounded-lg flex-none">
            <TabsTrigger value="from-apps" className="p-4">
              从已有应用选择
            </TabsTrigger>
            <TabsTrigger value="customized" className="p-4">
              自定义
            </TabsTrigger>
          </TabsList>
          <TabsContent value="from-apps" className="py-4">
            <div className="grid gap-4 py-1 px-6">
              <div className="flex">
                <Label htmlFor="kb_selection" className="w-[90px]">
                  应用选择
                </Label>
                <div className="pl-6 pr-6">
                  {kbs.length > 0 ? (
                    <DropdownMenu modal={true}>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="outline"
                          className="text-sm text-muted-foreground"
                        >
                          已选{botConfig?.kb_ids.length || 0}个，可多选 <ChevronDownIcon />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent className="w-56">
                        <DropdownMenuLabel>应用</DropdownMenuLabel>
                        <DropdownMenuSeparator />
                        {kbs.map((kb) => (
                          <DropdownMenuCheckboxItem
                            key={kb.id}
                            checked={botConfig.kb_ids.includes(kb.id)}
                            // onCheckedChange={(checked) =>
                            //   handleKbSelect(kb.id, kb.name, checked)
                            // }
                            onSelect={(e) => e.preventDefault()}
                          >
                            {kb.name}
                          </DropdownMenuCheckboxItem>
                        ))}
                      </DropdownMenuContent>
                    </DropdownMenu>
                  ) : (
                    <div>
                      <p className="text-sm text-muted-foreground">尚未配置应用</p>
                    </div>
                  )}
                </div>
                {selectedKbNames.length > 0 && (
                  <div className="flex gap-1.5 items-center">
                    {selectedKbNames.map((name) => (
                      <Badge variant="secondary" className="h-6" key={name}>
                        {name}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="customized" className="py-4">
            <div className="grid gap-4 py-1 px-6">
              <div className="flex">
                <Label htmlFor="basemodel" className="w-[90px]">
                  基模型选择 <span className="text-destructive">*</span>{' '}
                </Label>
                <div className="px-6">
                  {llms.length > 0 ? (
                    <Select
                      value={botConfig.model_id}
                      onValueChange={(value) =>
                        setBotConfig((prev) => ({
                          ...prev,
                          model_id: value,
                        }))
                      }
                    >
                      <SelectTrigger>
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
                    <div>
                      <p className="text-sm text-muted-foreground">尚未配置大模型</p>
                      <Button
                        variant="outline"
                        onClick={() => {
                          router.push('/config/model/llm');
                        }}
                      >
                        前往添加
                      </Button>
                    </div>
                  )}
                </div>
              </div>
              <div className="flex gap-6">
                <Label htmlFor="enable_search" className="w-[90px]">
                  启用联网搜索
                </Label>
                <Switch
                  id="enable_search"
                  checked={botConfig.enable_search}
                  onCheckedChange={(checked) => {
                    setBotConfig({
                      ...botConfig,
                      enable_search: checked,
                    });
                  }}
                />
              </div>
              <div className="flex gap-6">
                <Label htmlFor="enable_agent" className="w-[90px]">
                  Agentic模式
                </Label>
                <Switch
                  id="enable_agent"
                  checked={botConfig.enable_agent}
                  onCheckedChange={(checked) => {
                    setBotConfig({
                      ...botConfig,
                      enable_agent: checked,
                    });
                  }}
                />
              </div>
              <div className="flex">
                <Label htmlFor="kb_selection" className="w-[90px]">
                  知识库选择
                </Label>
                <div className="pl-6 pr-6">
                  {kbs.length > 0 ? (
                    <DropdownMenu modal={true}>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="outline"
                          className="text-sm text-muted-foreground"
                        >
                          已选{botConfig?.kb_ids.length || 0}个，可多选 <ChevronDownIcon />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent className="w-56">
                        <DropdownMenuLabel>知识库</DropdownMenuLabel>
                        <DropdownMenuSeparator />
                        {kbs.map((kb) => (
                          <DropdownMenuCheckboxItem
                            key={kb.id}
                            checked={botConfig.kb_ids.includes(kb.id)}
                            // onCheckedChange={(checked) =>
                            //   handleKbSelect(kb.id, kb.name, checked)
                            // }
                            onSelect={(e) => e.preventDefault()}
                          >
                            {kb.name}
                          </DropdownMenuCheckboxItem>
                        ))}
                      </DropdownMenuContent>
                    </DropdownMenu>
                  ) : (
                    <div>
                      <p className="text-sm text-muted-foreground">尚未配置知识库</p>
                    </div>
                  )}
                </div>
                {selectedKbNames.length > 0 && (
                  <div className="flex gap-1.5 items-center">
                    {selectedKbNames.map((name) => (
                      <Badge variant="secondary" className="h-6" key={name}>
                        {name}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
              <div className="flex">
                <Label htmlFor="mcp_selection" className="w-[90px]">
                  MCP选择
                </Label>
                <div className="pl-6 pr-6">
                  {mcps.length > 0 ? (
                    <DropdownMenu modal={true}>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="outline"
                          className="text-sm text-muted-foreground"
                        >
                          已选{botConfig.mcp_ids.length}个，可多选 <ChevronDownIcon />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent className="w-56">
                        <DropdownMenuLabel>MCP</DropdownMenuLabel>
                        <DropdownMenuSeparator />
                        {mcps.map((mcp) => (
                          <DropdownMenuCheckboxItem
                            key={mcp.id}
                            checked={botConfig.mcp_ids.includes(mcp.id)}
                            // onCheckedChange={(checked) =>
                            //   handleMcpSelect(mcp.id, mcp.name, checked)
                            // }
                            onSelect={(e) => e.preventDefault()}
                          >
                            {mcp.name}
                          </DropdownMenuCheckboxItem>
                        ))}
                      </DropdownMenuContent>
                    </DropdownMenu>
                  ) : (
                    <div>
                      <p className="text-sm text-muted-foreground">尚未配置MCP</p>
                    </div>
                  )}
                </div>
                {selectedMcpNames.length > 0 && (
                  <div className="flex gap-1.5 items-center">
                    {selectedMcpNames.map((name) => (
                      <Badge variant="secondary" className="h-6" key={name}>
                        {name}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            </div>

          </TabsContent>

        </Tabs>
      </div>
      {saveErrorMsg && (
        <Alert variant="destructive">
          <Terminal />
          <AlertTitle>{isCreate ? '创建应用失败' : '保存应用失败'}</AlertTitle>
          <AlertDescription>{saveErrorMsg}</AlertDescription>
        </Alert>
      )}
      <div className="pt-8 flex gap-6">
        <Button
          variant="secondary"
          className="w-20"
          onClick={() => {
            router.push('/apps');
          }}
        >
          取消
        </Button>

        <Button
          className="w-20"
        // onClick={() => {
        //   handleSaveChatConfig();
        // }}
        >
          {isCreate ? '创建' : '保存'}
        </Button>
      </div>
    </div>
  );
};
