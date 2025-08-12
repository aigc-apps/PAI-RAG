'use client';

import React, { useState, useEffect } from 'react';
import { Thread } from '@/components/assistant-ui/thread';
import ModelSelector from '@/components/model-selector/index';
import ToolUIWrapper from '@/components/assistant-ui/tool-ui';

export const Assistant = () => {
  // LLM 配置状态
  const [llmConfig, setLlmConfig] = useState({
    id: '',
    source: '',
    model_id: '',
  });
  const [optionsVisible, setoptionsVisible] = useState(true);

  // 页面加载时拉取 LLM 配置
  useEffect(() => {
    const fetchLLMConfig = async () => {
      try {
        const res = await fetch('/v1/config/llms');
        if (!res.ok) throw new Error('拉取 LLM 配置失败');
        const data = await res.json();
        const llms = data.data.items;
        if (llms.length > 0) setLlmConfig(llms[0]);
      } catch (error) {
        console.error('拉取 LLM 配置失败:', error);
      }
    };

    fetchLLMConfig();
  }, []);

  // 模型选择回调
  const handleModelChange = async (
    id: string,
    source: string,
    model_id: string,
  ) => {
    setLlmConfig((prev) => ({
      ...prev,
      id: id,
      source: source,
      model_id: model_id,
    }));

    setoptionsVisible(source !== 'chatbot');
    console.log(source);
  };

  const [selectedOptions, setSelectedOptions] = useState<string[]>([]);

  useEffect(() => {
    console.log('selectedOptions updated:', selectedOptions);
  }, [selectedOptions]);

  return (
    <div className="flex flex-col h-full">
      <header className="flex h-12 border-b">
        <div className="flex justify-start px-20 border-none">
          <ModelSelector
            selectedModel={{
              source: llmConfig.source || '',
              model_id: llmConfig.model_id || '',
            }}
            onModelChange={handleModelChange}
          />
        </div>
      </header>
      <Thread
        optionsVisible={optionsVisible}
        onToggleChange={(options) => {
          console.log('Received options from Thread:', options); // ✅ 添加日志
          setSelectedOptions(options); // 更新状态
        }}
      />
      <ToolUIWrapper />
    </div>
  );
};
