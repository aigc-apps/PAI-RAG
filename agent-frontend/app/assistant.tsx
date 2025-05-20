"use client";

import React, { useState, useEffect } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";
import ModelSelector from "@/components/model-selector/index";
import ToolUIWrapper from "@/components/assistant-ui/tool-ui";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import LlmConfig from "./config/llm/page";
import McpConfig from "./config/mcp/page";
import SearchConfig from "./config/search/page";


export const Assistant = () => {
  // LLM 配置状态
  const [llmConfig, setLlmConfig] = useState({
    id: Date.now(),
    source: "",
    model_name: "",
    api_key: "",
    max_context: 0,
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
  const handleModelChange = async (
    source: string,
    model_name: string,
    api_key: string,
  ) => {
    setLlmConfig({
      ...llmConfig,
      source,
      model_name,
      api_key,
    });
  };

  const [selectedOptions, setSelectedOptions] = useState<string[]>([]); // 存储 ToggleGroup 状态

  const runtime = useChatRuntime({
    // api: "/api/chat",
    api: `http://localhost:${process.env.NEXT_PUBLIC_BACKEND_PORT}/api/chat`,
    headers: {
      "X-Model-Name": llmConfig.model_name || "gpt-4o",
      "X-Api-Key": llmConfig.api_key || "",
      "X-Model-Source": llmConfig.source || "openai",
      "X-Options": selectedOptions.join(",") || "",
    },
  });

  const [activeTab, setActiveTab] = useState("/"); // 提升状态到父组件
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <SidebarProvider defaultOpen={true}>
        <AppSidebar activeTab={activeTab} setActiveTab={setActiveTab} /> 
        <SidebarInset className="h-screen overflow-hidden">
            {activeTab === "/" && 
              <div className="flex flex-col h-full">
                  <header className="flex h-12 border-b">
                  <SidebarTrigger />
                  <div className="flex justify-start px-20 border-none">
                    <ModelSelector
                      selectedModel={{
                        source: llmConfig.source || "",
                        model_name: llmConfig.model_name || ""
                      }}
                      onModelChange={handleModelChange}
                    />
                  </div>
                </header>
                <Thread onToggleChange={(options) => {
                    setSelectedOptions(options); // 更新状态
                  }}  />
                <ToolUIWrapper />
              </div>
            }
            {activeTab === "/config/llm" && 
              <div className="flex flex-col h-full">
                <header className="flex h-12 border-b">
                  <SidebarTrigger />
                </header>
                <LlmConfig />
              </div>
            }
            {activeTab === "/config/mcp" && 
              <div className="flex flex-col h-full">
                <header className="flex h-12 border-b">
                  <SidebarTrigger />
                </header>
                <McpConfig />
              </div>
            }
            {activeTab === "/config/search" && 
              <div className="flex flex-col h-full">
                <header className="flex h-12 border-b">
                  <SidebarTrigger />
                </header>
                <SearchConfig />
              </div>
            }
        </SidebarInset>
      </SidebarProvider>
    </AssistantRuntimeProvider>
  );
};
