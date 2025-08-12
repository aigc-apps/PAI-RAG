// contexts/ChatContext.tsx
// 管理聊天相关的options，用于在modal页面更新，聊天接口获取，
'use client';

import { createContext, useContext, useState, ReactNode } from 'react';

interface ChatOptions {
  enable_thinking: boolean;
  enable_search: boolean;
  mcp_ids: string[];           // mcp列表
  kb_ids: string[];            // 知识库列表

  updateEnableThinking: (thinking: boolean) => void;
  updateEnableSearch: (search: boolean) => void;
  updateMcpIds: (mcp_ids: string[]) => void;
  updateKbIds: (kb_ids: string[]) => void;
}

const ChatContext = createContext<ChatOptions | undefined>(undefined);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [enableThinking, setEnableThinking] = useState(false);
  const [enableSearch, setEnableSearch] = useState(false);
  const [mcpIds, setMcpIds] = useState<string[]>([]);
  const [kbIds, setKbIds] = useState<string[]>([]);

  const updateEnableThinking = (thinking: boolean) => {
    setEnableThinking(thinking);
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
        enable_thinking: enableThinking, 
        enable_search: enableSearch,
        mcp_ids: mcpIds,
        kb_ids: kbIds,
        updateEnableSearch,
        updateEnableThinking,
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