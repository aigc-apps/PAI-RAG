'use client';

import React from 'react';
import { cn } from '@/lib/utils';
import { useI18n } from '@/app/providers/i18n';

/**
 * Unified loading primitives for the app.
 *
 * - `Spinner` — the base animated indicator (three purple dots pulsing in
 *   sequence). Use directly when you need the dots without any label.
 * - `Loading` — spinner + optional label, three sizes. Pick this for
 *   inline list placeholders, dialog bodies, button states, etc.
 * - `PageLoading` — full-page centered layout (takes its parent height).
 *   Use for route-level "waiting for initial data" screens.
 * - `SkeletonLine` / `SkeletonCard` — shimmering placeholder rects for
 *   richer list / card skeletons.
 *
 * All primitives derive from the theme's `--primary` color so they stay
 * in sync with the purple accent.
 */

// ===== Spinner: three pulsing dots =====

const SIZE_DOT: Record<'sm' | 'md' | 'lg', string> = {
  sm: 'w-1 h-1',
  md: 'w-1.5 h-1.5',
  lg: 'w-2 h-2',
};

const SIZE_GAP: Record<'sm' | 'md' | 'lg', string> = {
  sm: 'gap-1',
  md: 'gap-1.5',
  lg: 'gap-2',
};

export function Spinner({
  size = 'md',
  className,
}: {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}) {
  const dot = SIZE_DOT[size];
  const gap = SIZE_GAP[size];
  return (
    <span
      role="status"
      aria-label="loading"
      className={cn('inline-flex items-center', gap, className)}
    >
      <span
        className={cn('rounded-full bg-primary animate-pulse-dot-1', dot)}
      />
      <span
        className={cn('rounded-full bg-primary animate-pulse-dot-2', dot)}
      />
      <span
        className={cn('rounded-full bg-primary animate-pulse-dot-3', dot)}
      />
    </span>
  );
}

// ===== Inline loading: spinner + label =====

const LABEL_SIZE: Record<'sm' | 'md' | 'lg', string> = {
  sm: 'text-[11px]',
  md: 'text-xs',
  lg: 'text-sm',
};

export function Loading({
  size = 'md',
  label,
  className,
}: {
  size?: 'sm' | 'md' | 'lg';
  label?: string;
  className?: string;
}) {
  const { t } = useI18n();
  const text = label ?? t('common.loading');
  return (
    <span
      className={cn(
        'inline-flex items-center gap-2 text-muted-foreground',
        LABEL_SIZE[size],
        className,
      )}
    >
      <Spinner size={size} />
      {text && <span>{text}</span>}
    </span>
  );
}

// ===== Full-page / full-area loading =====

export function PageLoading({
  label,
  className,
}: {
  label?: string;
  className?: string;
}) {
  const { t } = useI18n();
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3 py-16 w-full min-h-[240px]',
        className,
      )}
    >
      <Spinner size="lg" />
      <span className="text-xs text-muted-foreground">
        {label ?? t('common.loading')}
      </span>
    </div>
  );
}

// ===== Skeleton helpers =====

export function SkeletonLine({
  className,
  width,
}: {
  className?: string;
  width?: string | number;
}) {
  return (
    <div
      className={cn(
        'h-3 rounded-md bg-gradient-to-r from-muted via-muted/60 to-muted bg-[length:200%_100%] animate-shimmer',
        className,
      )}
      style={width !== undefined ? { width } : undefined}
    />
  );
}

export function SkeletonCard({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        'rounded-lg border border-border bg-card p-4 space-y-2.5',
        className,
      )}
    >
      <div className="flex items-center gap-2">
        <div className="w-7 h-7 rounded-md bg-gradient-to-r from-muted via-muted/60 to-muted bg-[length:200%_100%] animate-shimmer" />
        <SkeletonLine className="flex-1" />
      </div>
      <SkeletonLine width="80%" />
      <SkeletonLine width="60%" />
    </div>
  );
}
