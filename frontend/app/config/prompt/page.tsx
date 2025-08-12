"use client";

import { Button } from "@/components/ui/button";
import React, { useState, useEffect } from "react";
import { Label } from "@/components/ui/label";
import * as Toast from "@radix-ui/react-toast";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function PromptConfig() {
  const [systemPrompt, setSystemPrompt] = useState("");
  const [searchWebToolPrompt, setSearchWebToolPrompt] = useState("");
  const [planningToolPrompt, setPlanningToolPrompt] = useState("");
  const [attachmentsToolPrompt, setAttachmentsToolPrompt] = useState("");
  const [knowledgebaseToolPrompt, setKnowledgebaseToolPrompt] = useState("");
  const [withoutToolsPrompt, setWithoutToolsPrompt] = useState("");

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeTool, setActiveTool] = useState("search_web"); // 默认选中第一个工具
  const [toastState, setToastState] = useState({
    open: false,
    title: "",
    description: "",
    variant: "default" as "default" | "destructive",
  });

  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8688";

  // 工具列表配置
  const toolSections = [
    {
      id: "search_web",
      label: "Search Web Tool",
      state: searchWebToolPrompt,
      setter: setSearchWebToolPrompt,
    },
    {
      id: "planning",
      label: "Planning Tool",
      state: planningToolPrompt,
      setter: setPlanningToolPrompt,
    },
    {
      id: "attachments",
      label: "Attachments Tool",
      state: attachmentsToolPrompt,
      setter: setAttachmentsToolPrompt,
    },
    {
      id: "knowledgebase",
      label: "Knowledgebase Tool",
      state: knowledgebaseToolPrompt,
      setter: setKnowledgebaseToolPrompt,
    },
    {
      id: "without_tools",
      label: "Without Tools",
      state: withoutToolsPrompt,
      setter: setWithoutToolsPrompt,
    },
  ];

  // 获取当前激活的工具配置
  const activeToolConfig =
    toolSections.find((tool) => tool.id === activeTool) || toolSections[0];

  // 加载 prompt 配置
  useEffect(() => {
    const fetchPrompts = async () => {
      try {
        setIsLoading(true);
        const res = await fetch(`${API_BASE}/v1/config/prompts`, {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        });

        if (!res.ok) throw new Error("加载 Prompt 配置失败");

        const data = await res.json();
        const prompt = data.data.prompts || {};

        setSystemPrompt(prompt.system_prompt || "");
        setSearchWebToolPrompt(prompt.search_web_tool_prompt || "");
        setPlanningToolPrompt(prompt.planning_tool_prompt || "");
        setAttachmentsToolPrompt(prompt.attachments_tool_prompt || "");
        setKnowledgebaseToolPrompt(prompt.knowledgebase_tool_prompt || "");
        setWithoutToolsPrompt(prompt.without_tools_prompt || "");
      } catch (err: any) {
        setError(err.message || "加载失败");
        setToastState({
          open: true,
          title: "Prompt 加载失败",
          description: err.message || "请检查网络或重试",
          variant: "destructive",
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchPrompts();
  }, []);

  // 保存 prompt 配置
  const handleSave = async () => {
    try {
      setIsLoading(true);
      setError("");

      const data = {
        prompts: {
          system_prompt: systemPrompt,
          search_web_tool_prompt: searchWebToolPrompt,
          planning_tool_prompt: planningToolPrompt,
          attachments_tool_prompt: attachmentsToolPrompt,
          knowledgebase_tool_prompt: knowledgebaseToolPrompt,
          without_tools_prompt: withoutToolsPrompt,
        },
      };

      const res = await fetch(`${API_BASE}/v1/config/prompts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      if (!res.ok) throw new Error("保存失败，请检查网络或配置");

      setToastState({
        open: true,
        title: "Prompt 配置保存成功",
        description: "Prompt 配置已成功保存",
        variant: "default",
      });
    } catch (err: any) {
      setError(err.message || "保存失败，请重试");
      setToastState({
        open: true,
        title: "Prompt 配置保存失败",
        description: err.message || "请检查网络或重试",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div id="prompt-config" className="p-6 h-screen flex flex-col">
      <h1 className="text-2xl font-bold mb-4">Prompt 配置管理</h1>

      <div className="flex-1 flex gap-6 overflow-hidden">
        {/* 左侧 - System Prompt */}
        <div className="w-1/2 flex flex-col">
          <Card className="flex-1 flex flex-col h-full">
            <CardHeader>
              <CardTitle>System Prompt</CardTitle>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col overflow-hidden">
              <Textarea
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                className="flex-1 resize-none overflow-auto"
                placeholder="输入 System Prompt"
              />
            </CardContent>
          </Card>
        </div>

        {/* 右侧 - Tool Prompts */}
        <div className="w-1/2 flex flex-col">
          <Card className="flex-1 flex flex-col h-full">
            <CardHeader className="pb-0">
              <CardTitle className="text-lg">Tool Prompts</CardTitle>
            </CardHeader>

            {/* 工具选择标签 - 紧贴标题 */}
            <div className="px-6 py-2">
              <div className="flex flex-wrap gap-2">
                {toolSections.map((tool) => (
                  <button
                    key={tool.id}
                    onClick={() => setActiveTool(tool.id)}
                    className={`rounded-full px-3 py-2 text-sm transition-colors border ${
                      activeTool === tool.id
                        ? "bg-black text-white"
                        : "bg-white text-black hover:bg-gray-100 border-gray-200"
                    }`}
                  >
                    {tool.label}
                  </button>
                ))}
              </div>
            </div>

            {/* 工具 Prompt 编辑区域 */}
            <CardContent className="flex-1 flex flex-col pt-2 overflow-hidden">
              <Label htmlFor={activeToolConfig.id} className="mb-2">
                {activeToolConfig.label}
              </Label>
              <Textarea
                id={activeToolConfig.id}
                value={activeToolConfig.state}
                onChange={(e) => activeToolConfig.setter(e.target.value)}
                className="flex-1 resize-none overflow-auto"
                placeholder={`输入 ${activeToolConfig.label}`}
              />
            </CardContent>
          </Card>
        </div>
      </div>

      {/* 底部操作按钮 */}
      <div className="flex justify-end gap-4 mt-6">
        <Button onClick={handleSave} disabled={isLoading}>
          {isLoading ? "保存中..." : "保存所有 Prompt"}
        </Button>
        {error && <p className="text-red-500 self-center">{error}</p>}
      </div>

      {/* Toast 提示 */}
      <Toast.Provider>
        <Toast.Root
          open={toastState.open}
          onOpenChange={(open) => setToastState((prev) => ({ ...prev, open }))}
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
            onClick={() => setToastState((prev) => ({ ...prev, open: false }))}
          >
            ×
          </Toast.Action>
        </Toast.Root>
        <Toast.Viewport className="fixed bottom-0 right-0 z-[100] m-0 flex w-96 flex-col gap-2 p-6" />
      </Toast.Provider>
    </div>
  );
}
