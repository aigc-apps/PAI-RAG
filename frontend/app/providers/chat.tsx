// contexts/ChatContext.tsx
// 管理聊天相关的options，用于在modal页面更新，聊天接口获取，
'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

const CHAT_OPTIONS_STORAGE_KEY = 'pai-rag-chat-options';
const USER_QUERY_PARAM = 'user';

interface ChatOptionsState {
  model: string;
  enable_agent: boolean;
  enable_search: boolean;
  enable_chatdb: boolean;
  mcp_ids: string[];
  kb_ids: string[];
  user_id: string;
}

interface ChatOptions extends ChatOptionsState {
  updateModel: (model: string) => void;
  updateEnablePlanning: (planning: boolean) => void;
  updateEnableSearch: (search: boolean) => void;
  updateMcpIds: (mcp_ids: string[]) => void;
  updateKbIds: (kb_ids: string[]) => void;
  updateUser: (user_id: string) => void;
  updateEnableChatdb: (chatdb: boolean) => void;
}

const ChatContext = createContext<ChatOptions | undefined>(undefined);

// 默认值
const defaultChatOptions: ChatOptionsState = {
  model: '',
  enable_agent: false,
  enable_search: false,
  enable_chatdb: false,
  mcp_ids: [],
  kb_ids: [],
  user_id: '',
};

// 从 localStorage 读取状态
function loadChatOptionsFromStorage(): ChatOptionsState {
  // Testing override: ?user=xxx from URL wins over localStorage for user_id.
  let urlUser: string | undefined;
  try {
    urlUser = new URLSearchParams(window.location.search).get(USER_QUERY_PARAM)?.trim() || undefined;
  } catch {
    // ignore
  }

  const saved = localStorage.getItem(CHAT_OPTIONS_STORAGE_KEY);
  if (saved) {
    try {
      const parsed = JSON.parse(saved) as Partial<ChatOptionsState>;
      return {
        model: parsed.model ?? defaultChatOptions.model,
        enable_agent: parsed.enable_agent ?? defaultChatOptions.enable_agent,
        enable_search: parsed.enable_search ?? defaultChatOptions.enable_search,
        enable_chatdb: parsed.enable_chatdb ?? defaultChatOptions.enable_chatdb,
        mcp_ids: parsed.mcp_ids ?? defaultChatOptions.mcp_ids,
        kb_ids: parsed.kb_ids ?? defaultChatOptions.kb_ids,
        user_id: urlUser ?? parsed.user_id ?? defaultChatOptions.user_id,
      };
    } catch (e) {
      console.error('Failed to parse chat options from localStorage:', e);
    }
  }
  return {
    ...defaultChatOptions,
    user_id: urlUser ?? defaultChatOptions.user_id,
  };
}

export function ChatProvider({ children }: { children: ReactNode }) {
  // 使用默认值初始化，确保 SSR 和客户端首次渲染一致
  const [state, setState] = useState<ChatOptionsState>(defaultChatOptions);
  const [isHydrated, setIsHydrated] = useState(false);

  // 客户端挂载后从 localStorage 加载
  useEffect(() => {
    const loaded = loadChatOptionsFromStorage();
    setState(loaded);
    setIsHydrated(true);
  }, []);

  // 保存到 localStorage（仅在 hydrated 后）
  useEffect(() => {
    if (isHydrated) {
      localStorage.setItem(CHAT_OPTIONS_STORAGE_KEY, JSON.stringify(state));
    }
  }, [state, isHydrated]);

  const updateModel = (model: string) => {
    setState(prev => ({ ...prev, model }));
  };

  const updateEnablePlanning = (enable_agent: boolean) => {
    setState(prev => ({ ...prev, enable_agent }));
  };

  const updateEnableSearch = (enable_search: boolean) => {
    setState(prev => ({ ...prev, enable_search }));
  };

  const updateMcpIds = (mcp_ids: string[]) => {
    setState(prev => ({ ...prev, mcp_ids }));
  };

  const updateKbIds = (kb_ids: string[]) => {
    setState(prev => ({ ...prev, kb_ids }));
  };

  const updateUser = (user_id: string) => {
    setState(prev => ({ ...prev, user_id }));
  };

  const updateEnableChatdb = (enable_chatdb: boolean) => {
    setState(prev => ({ ...prev, enable_chatdb }));
  };

  return (
    <ChatContext.Provider value={{ 
        model: state.model,
        enable_agent: state.enable_agent, 
        enable_search: state.enable_search,
        mcp_ids: state.mcp_ids,
        kb_ids: state.kb_ids,
        user_id: state.user_id,
        enable_chatdb: state.enable_chatdb,
        updateModel,
        updateEnablePlanning,
        updateEnableSearch,
        updateMcpIds,
        updateKbIds,
        updateUser,
        updateEnableChatdb,
        }}>
      {children}
    </ChatContext.Provider>
  );
}

export function useChatOptions() {
  const context = useContext(ChatContext);
  if (!context) throw new Error('useChat must be used within ChatProvider');
  return context;
}