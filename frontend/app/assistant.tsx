'use client';

import React, { useState, useEffect } from 'react';
import { Thread } from '@/components/assistant-ui/thread';
import ModelSelector from '@/components/model-selector/index';
import ToolUIWrapper from '@/components/assistant-ui/tool-ui';
import { useChatOptions } from './providers/chat';

export const Assistant = () => {
  const [optionsVisible, setoptionsVisible] = useState(true);
  const {model, updateModel} = useChatOptions();

  // 模型选择回调
  const handleModelChange = async (
    id: string,
    source: string,
    model_id: string,
  ) => {

    updateModel(model_id);

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
              model_id: model || undefined,
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
