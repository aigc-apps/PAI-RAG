'use client';

import React, { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';

export const HEADER_SLOT_ID = 'app-header-slot';

/**
 * Renders its children into the global app header slot defined in layout.tsx.
 * Pages can use this to inject breadcrumbs, page actions, etc. without
 * leaving an empty band at the top of their content area.
 */
export function HeaderPortal({ children }: { children: React.ReactNode }) {
  const [target, setTarget] = useState<HTMLElement | null>(null);

  useEffect(() => {
    setTarget(document.getElementById(HEADER_SLOT_ID));
  }, []);

  if (!target) return null;
  return createPortal(children, target);
}
