'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

const TENANT_STORAGE_KEY = 'pai-rag-tenant-id';
const TENANTS_STORAGE_KEY = 'pai-rag-tenants';
const DEFAULT_TENANT_ID = '__default_tenant_id__';
const DEFAULT_TENANT_NAME = '默认空间';

interface Tenant {
  id: string;
  name: string;
}

interface TenantContextType {
  tenantId: string;
  tenantName: string;
  tenants: Tenant[];
  isHydrated: boolean;
  setTenant: (id: string, name: string) => void;
  addTenant: (id: string, name: string) => void;
  removeTenant: (id: string) => void;
}

const TenantContext = createContext<TenantContextType | undefined>(undefined);

// 默认状态（用于 SSR 和初始渲染）
const defaultState = {
  tenantId: DEFAULT_TENANT_ID,
  tenantName: DEFAULT_TENANT_NAME,
  tenants: [{ id: DEFAULT_TENANT_ID, name: DEFAULT_TENANT_NAME }] as Tenant[],
};

// 从 localStorage 读取状态的函数
function loadFromLocalStorage(): { tenantId: string; tenantName: string; tenants: Tenant[] } {
  const savedTenantId = localStorage.getItem(TENANT_STORAGE_KEY);
  const savedTenants = localStorage.getItem(TENANTS_STORAGE_KEY);

  let tenants: Tenant[] = [{ id: DEFAULT_TENANT_ID, name: DEFAULT_TENANT_NAME }];
  let tenantId = DEFAULT_TENANT_ID;
  let tenantName = DEFAULT_TENANT_NAME;

  if (savedTenants) {
    try {
      const parsedTenants = JSON.parse(savedTenants) as Tenant[];
      if (parsedTenants.length > 0) {
        tenants = parsedTenants;
        
        if (savedTenantId) {
          const found = parsedTenants.find(t => t.id === savedTenantId);
          if (found) {
            tenantId = found.id;
            tenantName = found.name;
          } else {
            tenantId = parsedTenants[0].id;
            tenantName = parsedTenants[0].name;
          }
        }
      }
    } catch (e) {
      console.error('Failed to parse tenants from localStorage:', e);
    }
  }

  return { tenantId, tenantName, tenants };
}

export function TenantProvider({ children }: { children: ReactNode }) {
  // 使用默认值初始化，确保 SSR 和客户端首次渲染一致
  const [state, setState] = useState(defaultState);
  const [isHydrated, setIsHydrated] = useState(false);
  const { tenantId, tenantName, tenants } = state;

  // 客户端挂载后从 localStorage 加载
  useEffect(() => {
    const loaded = loadFromLocalStorage();
    setState(loaded);
    setIsHydrated(true);
  }, []);

  // 保存到 localStorage（仅在 hydrated 后）
  useEffect(() => {
    if (isHydrated) {
      localStorage.setItem(TENANT_STORAGE_KEY, tenantId);
      localStorage.setItem(TENANTS_STORAGE_KEY, JSON.stringify(tenants));
    }
  }, [tenantId, tenants, isHydrated]);

  const setTenant = (id: string, name: string) => {
    setState(prev => ({ ...prev, tenantId: id, tenantName: name }));
  };

  const addTenant = (id: string, name: string) => {
    // 检查是否已存在
    if (tenants.some(t => t.id === id)) {
      return;
    }
    const newTenant = { id, name };
    setState(prev => ({
      tenantId: id,
      tenantName: name,
      tenants: [...prev.tenants, newTenant],
    }));
  };

  const removeTenant = (id: string) => {
    // 不允许删除默认工作空间
    if (id === DEFAULT_TENANT_ID) return;
    
    setState(prev => {
      const newTenants = prev.tenants.filter(t => t.id !== id);
      // 如果删除的是当前 tenant，切换到默认
      if (prev.tenantId === id) {
        return {
          tenantId: DEFAULT_TENANT_ID,
          tenantName: DEFAULT_TENANT_NAME,
          tenants: newTenants,
        };
      }
      return { ...prev, tenants: newTenants };
    });
  };

  return (
    <TenantContext.Provider value={{
      tenantId,
      tenantName,
      tenants,
      isHydrated,
      setTenant,
      addTenant,
      removeTenant,
    }}>
      {children}
    </TenantContext.Provider>
  );
}

export function useTenant() {
  const context = useContext(TenantContext);
  if (!context) {
    throw new Error('useTenant must be used within TenantProvider');
  }
  return context;
}

/**
 * 创建带有 X-TENANT-ID header 的 fetch 函数
 * 用于替代原生 fetch，自动添加工作空间 header
 */
export function createTenantFetch(tenantId: string) {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const headers = new Headers(init?.headers);
    headers.set('X-TENANT-ID', tenantId);
    
    return fetch(input, {
      ...init,
      headers,
    });
  };
}

/**
 * 获取 tenant header 对象，用于添加到现有 headers
 */
export function getTenantHeader(tenantId: string): Record<string, string> {
  return {
    'X-TENANT-ID': tenantId,
  };
}

