"use client";

import React, { useState, useEffect } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";
import { ThreadList } from "@/components/assistant-ui/thread-list";
import ModelSelector from "@/components/model-selector/index";
import ToolUIWrapper from "@/components/assistant-ui/tool-ui";
import { AppSidebar } from "@/components/app-sidebar"
import { SidebarProvider, SidebarTrigger, SidebarInset } from "@/components/ui/sidebar"

export const Assistant = () => {
  // LLM 配置状态
  const [llmConfig, setLlmConfig] = useState({
    id: Date.now(),
    source: "",
    model_name: "",
    api_key: "",
    max_context: 0
  });

  // 页面加载时拉取 LLM 配置
  useEffect(() => {
    const fetchLLMConfig = async () => {
      try {
        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8097;
        console.log("assistant BACKEND_PORT", port);
        const res = await fetch(`http://localhost:${port}/api/configs`);
        if (!res.ok) throw new Error("拉取 LLM 配置失败");
        const data = await res.json();
        if (data.llm_config?.[0]) setLlmConfig(data.llm_config[0]);
      } catch (error) {
        console.error("拉取 LLM 配置失败:", error);
      }
    };

    fetchLLMConfig();
  }, []);

  // 模型选择回调
  const handleModelChange = async (source: string, model_name: string, api_key: string) => {
    setLlmConfig({
      ...llmConfig,
      source,
      model_name,
      api_key
    });
  };

  const [selectedOptions, setSelectedOptions] = useState<string[]>([]); // 存储 ToggleGroup 状态

  const runtime = useChatRuntime({
    api: "/api/chat",
    headers: { "X-Model-Name": llmConfig.model_name || "gpt-4o" , "X-Api-Key": llmConfig.api_key || "" , "X-Model-Source": llmConfig.source || "openai" ,"X-Options": selectedOptions.join(",") || ""},
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <SidebarProvider defaultOpen={false}>
        <AppSidebar />
        <SidebarInset className="h-full h-screen !overflow-hidden">
          <header className="flex h-12">
          <SidebarTrigger />
            <div className="flex justify-start px-60">
              <ModelSelector
                selectedModel={{
                  source: llmConfig.source || "",
                  model_name: llmConfig.model_name || ""
                }}
                onModelChange={handleModelChange}
              />
            </div>
          
          </header>
          <div className="grid grid-cols-[200px_1fr_auto] h-[calc(100%-3rem)] !overflow-hidden">
            <ThreadList />
            <Thread onToggleChange={(options) => {
                setSelectedOptions(options); // 更新状态
              }}  />
            <ToolUIWrapper />
          </div>
        </SidebarInset>
    </SidebarProvider>
    </AssistantRuntimeProvider>
  );
};
