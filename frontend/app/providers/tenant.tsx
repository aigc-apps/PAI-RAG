'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

const DEFAULT_TENANT_ID = '__default_tenant_id__';
const DEFAULT_TENANT_NAME = 'default workspace';
const WORKSPACE_QUERY_PARAM = 'workspace';

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

const defaultState = {
  tenantId: DEFAULT_TENANT_ID,
  tenantName: DEFAULT_TENANT_NAME,
  tenants: [{ id: DEFAULT_TENANT_ID, name: DEFAULT_TENANT_NAME }] as Tenant[],
};

/**
 * Read the workspace from the URL query string.
 * `?workspace=xxx` — use xxx as tenant id.
 * Empty / missing — fall back to default.
 */
function loadFromUrl(): { tenantId: string; tenantName: string; tenants: Tenant[] } {
  const params = new URLSearchParams(window.location.search);
  const workspace = params.get(WORKSPACE_QUERY_PARAM)?.trim();

  if (workspace) {
    return {
      tenantId: workspace,
      tenantName: workspace,
      tenants: [
        { id: DEFAULT_TENANT_ID, name: DEFAULT_TENANT_NAME },
        { id: workspace, name: workspace },
      ],
    };
  }

  return {
    tenantId: DEFAULT_TENANT_ID,
    tenantName: DEFAULT_TENANT_NAME,
    tenants: [{ id: DEFAULT_TENANT_ID, name: DEFAULT_TENANT_NAME }],
  };
}

export function TenantProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState(defaultState);
  const [isHydrated, setIsHydrated] = useState(false);
  const { tenantId, tenantName, tenants } = state;

  // Hydrate from URL after mount (SSR compatibility)
  useEffect(() => {
    const loaded = loadFromUrl();
    setState(loaded);
    setIsHydrated(true);
  }, []);

  const setTenant = (id: string, name: string) => {
    setState(prev => ({ ...prev, tenantId: id, tenantName: name }));
  };

  const addTenant = (id: string, name: string) => {
    if (tenants.some(t => t.id === id)) return;
    const newTenant = { id, name };
    setState(prev => ({
      tenantId: id,
      tenantName: name,
      tenants: [...prev.tenants, newTenant],
    }));
  };

  const removeTenant = (id: string) => {
    if (id === DEFAULT_TENANT_ID) return;
    setState(prev => {
      const newTenants = prev.tenants.filter(t => t.id !== id);
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
 * Helper: create a fetch wrapper that injects X-TENANT-ID header.
 */
export function createTenantFetch(tenantId: string) {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const headers = new Headers(init?.headers);
    headers.set('X-TENANT-ID', tenantId);
    return fetch(input, { ...init, headers });
  };
}

export function getTenantHeader(tenantId: string): Record<string, string> {
  return { 'X-TENANT-ID': tenantId };
}
