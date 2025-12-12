/**
 * 创建带有 X-TENANT-ID header 的 fetch wrapper
 * 用于在页面组件中发送带工作空间 ID 的请求
 */

/**
 * 创建带有工作空间 ID header 的 fetch 函数
 * @param tenantId 工作空间 ID
 * @returns 包装后的 fetch 函数
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
 * @param tenantId 工作空间 ID  
 * @returns header 对象
 */
export function getTenantHeaders(tenantId: string): Record<string, string> {
  return {
    'X-TENANT-ID': tenantId,
  };
}

/**
 * 合并现有 headers 和 tenant header
 * @param tenantId 工作空间 ID
 * @param existingHeaders 现有的 headers
 * @returns 合并后的 headers 对象
 */
export function mergeWithTenantHeaders(
  tenantId: string, 
  existingHeaders?: HeadersInit
): Record<string, string> {
  const tenantHeaders = getTenantHeaders(tenantId);
  
  if (!existingHeaders) {
    return tenantHeaders;
  }
  
  if (existingHeaders instanceof Headers) {
    const result: Record<string, string> = { ...tenantHeaders };
    existingHeaders.forEach((value, key) => {
      result[key] = value;
    });
    return result;
  }
  
  if (Array.isArray(existingHeaders)) {
    const result: Record<string, string> = { ...tenantHeaders };
    existingHeaders.forEach(([key, value]) => {
      result[key] = value;
    });
    return result;
  }
  
  return {
    ...tenantHeaders,
    ...existingHeaders,
  };
}

