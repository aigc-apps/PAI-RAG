'use client';

import { useTenant } from '@/app/providers/tenant';
import { createTenantFetch, getTenantHeaders, mergeWithTenantHeaders } from '@/lib/tenant-fetch';
import { useCallback, useMemo, useRef } from 'react';

/**
 * 自定义 Hook：返回带有工作空间 ID 的 fetch 函数和辅助方法
 * 
 * tenantFetch 会自动等待 hydration 完成后再发送请求，确保使用正确的 tenantId
 * 
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { tenantFetch, tenantHeaders, tenantId, isHydrated } = useTenantFetch();
 * 
 *   // 使用 tenantFetch 发送请求（自动带上 X-TENANT-ID，会等待 hydration）
 *   const fetchData = async () => {
 *     const res = await tenantFetch('/api/data');
 *     return res.json();
 *   };
 * 
 *   // 或者使用 tenantHeaders 添加到现有请求
 *   const fetchWithHeaders = async () => {
 *     const res = await fetch('/api/data', {
 *       headers: {
 *         'Content-Type': 'application/json',
 *         ...tenantHeaders,
 *       },
 *     });
 *     return res.json();
 *   };
 * }
 * ```
 */
export function useTenantFetch() {
  const { tenantId, isHydrated } = useTenant();
  
  // 使用 ref 存储最新的 tenantId 和 isHydrated 状态
  const stateRef = useRef({ tenantId, isHydrated });
  stateRef.current = { tenantId, isHydrated };

  // 创建带工作空间 ID 的 fetch 函数，会等待 hydration 完成
  const tenantFetch = useMemo(() => {
    return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      // 等待 hydration 完成
      while (!stateRef.current.isHydrated) {
        await new Promise(resolve => setTimeout(resolve, 10));
      }
      
      // 使用最新的 tenantId
      const headers = new Headers(init?.headers);
      headers.set('X-TENANT-ID', stateRef.current.tenantId);
      
      return fetch(input, {
        ...init,
        headers,
      });
    };
  }, []); // 不依赖 tenantId，通过 ref 获取最新值

  // 获取工作空间 headers
  const tenantHeaders = useMemo(() => {
    return getTenantHeaders(tenantId);
  }, [tenantId]);

  // 合并 headers 的辅助函数
  const withTenantHeaders = useCallback((existingHeaders?: HeadersInit) => {
    return mergeWithTenantHeaders(tenantId, existingHeaders);
  }, [tenantId]);

  return {
    tenantId,
    isHydrated,
    tenantFetch,
    tenantHeaders,
    withTenantHeaders,
  };
}

