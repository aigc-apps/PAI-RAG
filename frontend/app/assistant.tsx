'use client';

import React, { useState, useEffect } from 'react';
import { Thread } from '@/components/assistant-ui/thread';
import ModelSelector from '@/components/model-selector/index';
import UserIdInput from '@/components/user/index';
import ToolUIWrapper from '@/components/assistant-ui/tool-ui';
import { useChatOptions } from './providers/chat';

export const Assistant = () => {
  const [optionsVisible, setoptionsVisible] = useState(true);
  const {
    model, 
    updateModel, 
    user_id, 
    updateUser,
    updateEnablePlanning,
    updateEnableSearch,
    updateMcpIds,
    updateKbIds,
    updateEnableChatdb,
  } = useChatOptions();

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

  const handleUserChange = async (user_id: string) => {
    updateUser(user_id);
  }

  const [selectedOptions, setSelectedOptions] = useState<string[]>([]);

  useEffect(() => {
    console.log('selectedOptions updated:', selectedOptions);
  }, [selectedOptions]);

  return (
    <div className="flex flex-col h-screen">
      <header className="flex p-1 border-b">
        <ModelSelector
          selectedModel={{
            model_id: model || undefined,
          }}
          onModelChange={handleModelChange}
        />
        <UserIdInput
          user_id={user_id}
          onChange={handleUserChange}
        />
      </header>
      <div className="flex flex-col flex-1 justify-end pb-8 overflow-y-auto">
        <Thread
          optionsVisible={optionsVisible}
          onToggleChange={(options) => {
            console.log('Received options from Thread:', options); // ✅ 添加日志
            setSelectedOptions(options); // 更新状态
          }}
        />
        <ToolUIWrapper />
      </div>
    </div>
  );
};
