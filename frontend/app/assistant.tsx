'use client';

import React, { useState } from 'react';
import { Thread } from '@/components/assistant-ui/thread';
import ModelSelector from '@/components/model-selector/index';
import ToolUIWrapper from '@/components/assistant-ui/tool-ui';
import { useChatOptions } from './providers/chat';
import { HeaderPortal } from '@/components/header-portal';

export const Assistant = () => {
  const [optionsVisible, setoptionsVisible] = useState(true);
  const { model, updateModel } = useChatOptions();

  // 模型选择回调
  const handleModelChange = async (
    _id: string,
    source: string,
    model_id: string,
  ) => {
    updateModel(model_id);
    setoptionsVisible(source !== 'chatbot');
  };

  const [, setSelectedOptions] = useState<string[]>([]);

  return (
    <div className="flex flex-col h-full min-h-0">
      <HeaderPortal>
        <ModelSelector
          selectedModel={{
            model_id: model || undefined,
          }}
          onModelChange={handleModelChange}
        />
      </HeaderPortal>
      <div className="flex flex-col flex-1 justify-end pb-8 overflow-y-auto">
        <Thread
          optionsVisible={optionsVisible}
          onToggleChange={(options) => {
            setSelectedOptions(options);
          }}
        />
        <ToolUIWrapper />
      </div>
    </div>
  );
};
