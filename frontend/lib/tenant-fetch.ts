/**
 * Create fetch wrapper with X-TENANT-ID and Accept-Language headers
 * Used for sending requests with workspace ID and locale in page components
 */

import { i18n, Language } from './i18n';

/**
 * Create fetch function with workspace ID and locale headers
 * @param tenantId Workspace ID
 * @param locale Optional locale override (defaults to current i18n language)
 * @returns Wrapped fetch function
 */
export function createTenantFetch(tenantId: string, locale?: Language) {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const headers = new Headers(init?.headers);
    headers.set('X-TENANT-ID', tenantId);
    
    // Add Accept-Language header with current locale
    const currentLocale = locale || i18n.getLanguage();
    headers.set('Accept-Language', currentLocale);
    
    return fetch(input, {
      ...init,
      headers,
    });
  };
}

/**
 * Get tenant and locale headers object for adding to existing headers
 * @param tenantId Workspace ID
 * @param locale Optional locale override (defaults to current i18n language)
 * @returns Header object
 */
export function getTenantHeaders(tenantId: string, locale?: Language): Record<string, string> {
  const currentLocale = locale || i18n.getLanguage();
  return {
    'X-TENANT-ID': tenantId,
    'Accept-Language': currentLocale,
  };
}

/**
 * Merge existing headers with tenant and locale headers
 * @param tenantId Workspace ID
 * @param existingHeaders Existing headers
 * @param locale Optional locale override (defaults to current i18n language)
 * @returns Merged headers object
 */
export function mergeWithTenantHeaders(
  tenantId: string, 
  existingHeaders?: HeadersInit,
  locale?: Language
): Record<string, string> {
  const tenantHeaders = getTenantHeaders(tenantId, locale);
  
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

