// contexts/ChatContext.tsx
// 管理聊天相关的options，用于在modal页面更新，聊天接口获取，
'use client';

import { createContext, useContext, useState, ReactNode } from 'react';

interface ChatOptions {
  model: string;
  enable_agent: boolean;
  enable_search: boolean;
  mcp_ids: string[];           // mcp列表
  kb_ids: string[];            // 知识库列表

  updateModel: (model: string) => void;
  updateEnablePlanning: (planning: boolean) => void;
  updateEnableSearch: (search: boolean) => void;
  updateMcpIds: (mcp_ids: string[]) => void;
  updateKbIds: (kb_ids: string[]) => void;
}

const ChatContext = createContext<ChatOptions | undefined>(undefined);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [model, setModel] = useState('');
  const [enablePlanning, setEnablePlanning] = useState(false);
  const [enableSearch, setEnableSearch] = useState(false);
  const [mcpIds, setMcpIds] = useState<string[]>([]);
  const [kbIds, setKbIds] = useState<string[]>([]);

  const updateModel = (model: string) => {
    setModel(model);
  }

  const updateEnablePlanning = (planning: boolean) => {
    setEnablePlanning(planning);
  }
  const updateEnableSearch = (search: boolean) => {
    setEnableSearch(search);
  }
  const updateMcpIds = (mcp_ids: string[]) => {
    setMcpIds(mcp_ids);
  }
  const updateKbIds = (kb_ids: string[]) => {
    setKbIds(kb_ids);
  }

  return (
    <ChatContext.Provider value={{ 
        model: model,
        enable_agent: enablePlanning, 
        enable_search: enableSearch,
        mcp_ids: mcpIds,
        kb_ids: kbIds,
        updateModel,
        updateEnableSearch,
        updateEnablePlanning,
        updateMcpIds,
        updateKbIds}}>
      {children}
    </ChatContext.Provider>
  );
}

export function useChatOptions() {
  const context = useContext(ChatContext);
  if (!context) throw new Error('useChat must be used within ChatProvider');
  return context;
}