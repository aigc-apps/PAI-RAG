"use client";
import React, { useState, useEffect, FC } from "react";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { ChevronDownIcon, Terminal } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

import { Button } from "@/components/ui/button";
import { MCPConfig } from "@/app/config/mcp/page";
import { LlmConfig } from "@/app/config/model/llm/page";
import { KbConfig } from "@/app/knowledgebase/kbconfig";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";

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
  setActiveTab: (tab: string) => void;
}

const default_chat_config = {
  id: "",
  app_id: "",
  description: "",
  enable_search: false,
  mcp_ids: [],
  kb_ids: [],
  enable_agent: false,
  model_id: "",
  updated_at: "",
};

// 知识库配置卡片
export const ChatbotConfigCard: FC<ChatbotConfigProps> = ({
  chatbotId,
  setActiveTab,
}) => {
  const [botConfig, setBotConfig] = useState<Chatbot>(default_chat_config);
  const [llms, setLlms] = useState<LlmConfig[]>([]);
  const [mcps, setMcps] = useState<MCPConfig[]>([]);
  const [kbs, setKbs] = useState<KbConfig[]>([]);
  const [selectedKbNames, setSelectedKbNames] = useState<string[]>([]);
  const [selectedMcpNames, setSelectedMcpNames] = useState<string[]>([]);
  const [saveErrorMsg, setSaveErrorMsg] = useState("");
  const isCreate: boolean = chatbotId === undefined || chatbotId === "";
  // const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const [llmRes] = await Promise.all([fetch("/v1/config/llms")]);

        const llmData = (await llmRes.json())?.data.items || [];
        console.log("llmData", llmData);
        setLlms([...llmData]);

        const [mcpRes] = await Promise.all([fetch("/v1/config/mcps")]);

        const mcpData =
          ((await mcpRes.json())?.data.items as MCPConfig[]) || [];
        console.log("mcpData", mcpData);
        setMcps([...mcpData]);

        const [kbRes] = await Promise.all([fetch("/v1/config/knowledgebases")]);

        const kbData = ((await kbRes.json())?.data.items as KbConfig[]) || [];
        console.log("kbData", kbData);
        setKbs([...kbData]);

        if (!isCreate) {
          const botRes = await fetch(`/v1/config/chatbots?app_id=${chatbotId}`);
          const botData = await botRes.json();
          setBotConfig(botData.data);
          console.log("chatbotData: ", botData.data);

          const kbnames = kbData
            .filter((item) => botData.data.kb_ids.includes(item.id))
            .map((item) => item.name);
          setSelectedKbNames([...kbnames]);
          console.log("selectedKbNames", kbnames);

          const mcpnames = mcpData
            .filter((item) => botData.data.mcp_ids.includes(item.id))
            .map((item) => item.name);
          setSelectedMcpNames([...mcpnames]);
          console.log("selectedMcpNames", mcpnames);
        }
      } catch (err: unknown) {
        console.log(err || "加载失败");
      }
    };
    fetchModelConfigs();
  }, [chatbotId, isCreate]);

  const handleSaveChatConfig = async () => {
    console.log("保存应用结果:", botConfig);
    const submit_url = isCreate
      ? "/v1/config/chatbots"
      : `/v1/config/chatbots/${botConfig.id}`;
    const updateMethod = isCreate ? "POST" : "PATCH";
    try {
      const res = await fetch(submit_url, {
        method: updateMethod,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(botConfig), // 包装为数组
      });

      if (!res.ok) throw new Error(`保存应用失败: ${await res.text()}`);
      setActiveTab("/chatbot");
      setSaveErrorMsg("");
      // onSaveSuccess(jsondata.data as KbConfig);
    } catch (err: any) {
      console.log("保存应用失败", err.message);
      setSaveErrorMsg(err.message);
    }
  };

  const handleKbSelect = (kb_id: string, kb_name: string, checked: boolean) => {
    console.log("handleKbSelect", kb_id, kb_name, checked);
    if (checked) {
      const kb_ids = botConfig.kb_ids.includes(kb_id)
        ? botConfig.kb_ids
        : [...botConfig.kb_ids, kb_id];
      setBotConfig((prev) => ({ ...prev, kb_ids: kb_ids }));
      if (!selectedKbNames.includes(kb_name)) {
        setSelectedKbNames((prev) => [...prev, kb_name]);
      }
    } else {
      const kb_ids = botConfig.kb_ids.filter((id) => id !== kb_id);
      setBotConfig((prev) => ({ ...prev, kb_ids: kb_ids }));
      if (selectedKbNames.includes(kb_name)) {
        setSelectedKbNames((prev) => prev.filter((name) => name !== kb_name));
      }
    }
  };
  const handleMcpSelect = (
    mcp_id: string,
    mcp_name: string,
    checked: boolean,
  ) => {
    if (checked) {
      const mcp_ids = botConfig.mcp_ids.includes(mcp_id)
        ? botConfig.mcp_ids
        : [...botConfig.mcp_ids, mcp_id];
      setBotConfig((prev) => ({ ...prev, mcp_ids: mcp_ids }));
      if (!selectedMcpNames.includes(mcp_name)) {
        setSelectedMcpNames((prev) => [...prev, mcp_name]);
      }
    } else {
      const mcp_ids = botConfig.mcp_ids.filter((id) => id !== mcp_id);
      setBotConfig((prev) => ({ ...prev, mcp_ids: mcp_ids }));
      if (selectedMcpNames.includes(mcp_name)) {
        setSelectedMcpNames((prev) => prev.filter((name) => name !== mcp_name));
      }
    }
  };

  return (
    <div className="grid gap-4 py-6 px-6">
      <div className="text-xl font-bold">
        {isCreate ? "新建应用" : "编辑应用"}
      </div>
      <div className="space-y-2">
        <Label htmlFor="app-id">
          App ID <span className="text-destructive">*</span>
        </Label>
        <Input
          id="appid"
          value={botConfig.app_id}
          onChange={(e) =>
            setBotConfig((prev) => ({ ...prev, app_id: e.target.value }))
          }
          placeholder="请输入应用ID, 如chatbot"
          required
        />
        <p className="text-sm text-muted-foreground">
          可输入大小写字母和数字,必须字母开头,3-64个字符。
        </p>
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
          placeholder="应用描述（可选）"
          rows={3}
        />
      </div>
      <div className="flex">
        <Label htmlFor="basemodel" className="w-[90px]">
          基模型选择 <span className="text-destructive">*</span>{" "}
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
              <p className="text-sm text-muted-foreground">尚未配置大模型，</p>
              <Button
                variant="outline"
                onClick={() => {
                  setActiveTab("/config/model/llm");
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
        <Label htmlFor="enable_search" className="w-[90px]">
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
                  已选{botConfig.kb_ids.length}个，可多选 <ChevronDownIcon />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56">
                <DropdownMenuLabel>知识库</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {kbs.map((kb) => (
                  <DropdownMenuCheckboxItem
                    key={kb.id}
                    checked={botConfig.kb_ids.includes(kb.id)}
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
      {saveErrorMsg && (
        <Alert variant="destructive">
          <Terminal />
          <AlertTitle>{isCreate ? "创建应用失败" : "保存应用失败"}</AlertTitle>
          <AlertDescription>{saveErrorMsg}</AlertDescription>
        </Alert>
      )}
      <div className="pt-8 flex gap-6">
        <Button
          variant="secondary"
          className="w-20"
          onClick={() => {
            setActiveTab("/chatbot");
          }}
        >
          取消
        </Button>

        <Button
          className="w-20"
          onClick={() => {
            handleSaveChatConfig();
          }}
        >
          {isCreate ? "创建" : "保存"}
        </Button>
      </div>
    </div>
  );
};
