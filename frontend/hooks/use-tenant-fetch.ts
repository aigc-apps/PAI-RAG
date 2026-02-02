'use client';

import { useTenant } from '@/app/providers/tenant';
import { createTenantFetch, getTenantHeaders, mergeWithTenantHeaders } from '@/lib/tenant-fetch';
import { i18n } from '@/lib/i18n';
import { useCallback, useMemo, useRef } from 'react';

/**
 * Custom Hook: Returns fetch function and helper methods with workspace ID and locale headers
 * 
 * tenantFetch will automatically wait for hydration to complete before sending requests, ensuring correct tenantId and locale
 * 
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { tenantFetch, tenantHeaders, tenantId, isHydrated } = useTenantFetch();
 * 
 *   // Use tenantFetch to send request (automatically includes X-TENANT-ID and Accept-Language, waits for hydration)
 *   const fetchData = async () => {
 *     const res = await tenantFetch('/api/data');
 *     return res.json();
 *   };
 * 
 *   // Or use tenantHeaders to add to existing request
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
  
  // Use ref to store latest tenantId, isHydrated, and locale state
  const stateRef = useRef({ tenantId, isHydrated, locale: i18n.getLanguage() });
  stateRef.current = { tenantId, isHydrated, locale: i18n.getLanguage() };

  // Create fetch function with workspace ID and locale, waits for hydration to complete
  const tenantFetch = useMemo(() => {
    return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      // Wait for hydration to complete
      while (!stateRef.current.isHydrated) {
        await new Promise(resolve => setTimeout(resolve, 10));
      }
      
      // Use latest tenantId and locale
      const headers = new Headers(init?.headers);
      headers.set('X-TENANT-ID', stateRef.current.tenantId);
      headers.set('Accept-Language', stateRef.current.locale);
      
      return fetch(input, {
        ...init,
        headers,
      });
    };
  }, []); // No dependency on tenantId or locale, get latest value through ref

  // Get workspace and locale headers
  const tenantHeaders = useMemo(() => {
    return getTenantHeaders(tenantId);
  }, [tenantId]);

  // Helper function to merge headers
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

