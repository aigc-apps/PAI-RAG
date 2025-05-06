
import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  SaveIcon,
} from "lucide-react";
import LLMConfig from "./llm-config";
import MCPConfig from "./mcp-config";
export default function ConfigPage() {
    const [llmConfig, setLlmConfig] = useState([{
      id: 1, 
      source: "",
      model_name: "",
      api_key: "",
      max_context: 1024,
    }]);
    const [mcpConfig, setMCPConfig] = useState([
      { id: 1, name: "", url: "", type: "", active: false },
    ]);

    const [isLoading, setIsLoading] = useState(true);
    // 从 API 加载配置
    useEffect(() => {
      const fetchConfig = async () => {
        setIsLoading(true);
        try {
          const res = await fetch("http://localhost:8097/api/configs");
          if (!res.ok) throw new Error("获取配置失败");
          const data = await res.json();
          setLlmConfig(data["llm_config"]);
          setMCPConfig(data["mcp_config"])
        } catch (err) {
          console.error(err);
          // 可选：设置默认值或提示用户
        } finally {
          setIsLoading(false);
        }
      };

      fetchConfig();
    }, []);

    const handleSave = async () => {
      // 保存逻辑（如 localStorage 或 API 请求）
      try {
        const res = await fetch("http://localhost:8097/api/configs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ llm_config: llmConfig, mcp_config: mcpConfig}),
          });
          if (!res.ok) throw new Error("保存失败");
          alert("配置已保存至本地文件");
        } catch (err) {
          console.error(err);
          alert("保存失败，请重试");
        }
    };

    return (
      <div className="space-y-4 p-4 overflow-y-auto max-h-screen">
        {isLoading ? (
        <p>加载中...</p>
      ) : (
        <>
          
          <LLMConfig
            config={llmConfig}
            onChange={(updatedConfig) => setLlmConfig(updatedConfig)}
          />
          <MCPConfig
            config={mcpConfig}
            onChange={(updatedConfig) => setMCPConfig(updatedConfig)}
          />
          <Button onClick={handleSave} className="w-full">
            <SaveIcon className="w-4 h-4 mr-2" />
            保存配置
          </Button>
        </>
      )}
      </div>
    )
  }