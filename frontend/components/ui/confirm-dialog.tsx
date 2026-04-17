'use client';

import React, { useEffect } from 'react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Spinner } from '@/components/ui/loading';
import { AlertTriangle, Trash2, Info } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useI18n } from '@/app/providers/i18n';

type Variant = 'destructive' | 'warning' | 'info';

const VARIANT_CONFIG: Record<
  Variant,
  {
    icon: React.ReactNode;
    iconBg: string;
    iconColor: string;
    confirmClass: string;
  }
> = {
  destructive: {
    icon: <Trash2 className="w-4 h-4" />,
    iconBg: 'bg-destructive/10',
    iconColor: 'text-destructive',
    confirmClass:
      'bg-destructive text-destructive-foreground hover:bg-destructive/90',
  },
  warning: {
    icon: <AlertTriangle className="w-4 h-4" />,
    iconBg: 'bg-amber-500/10',
    iconColor: 'text-amber-600',
    confirmClass: 'bg-amber-600 text-white hover:bg-amber-700',
  },
  info: {
    icon: <Info className="w-4 h-4" />,
    iconBg: 'bg-primary/10',
    iconColor: 'text-primary',
    confirmClass: '',
  },
};

export interface ConfirmDialogProps {
  /** Controlled open state. */
  open: boolean;
  /** Called with `false` when the user closes via ESC / overlay / cancel. */
  onOpenChange: (open: boolean) => void;

  /** Visual style. Defaults to `destructive` since most confirms are deletes. */
  variant?: Variant;
  /** Override the default variant icon. */
  icon?: React.ReactNode;

  /** Main heading, e.g. "是否确认删除?". */
  title: React.ReactNode;
  /** Body text or node. Use for descriptive copy. */
  description?: React.ReactNode;
  /** Extra custom content rendered between description and footer. */
  children?: React.ReactNode;

  /** Highlight a target name/id block (name/label row). */
  target?: {
    label?: string;
    value: string;
  };

  /** Footer button labels — defaults to i18n cancel/delete. */
  confirmLabel?: string;
  cancelLabel?: string;

  /** Confirm click. The dialog does NOT auto-close; caller controls via `open`. */
  onConfirm: () => void | Promise<void>;

  /** When true, confirm button shows a spinner and is disabled. */
  loading?: boolean;
  /** Disable the confirm button without showing a spinner. */
  disabled?: boolean;
}

/**
 * Unified confirmation dialog used across delete / warning / info flows.
 *
 * The body uses the same three-segment structure as KbModal / McpModal
 * (bordered header, body, muted footer), so confirms feel like part of
 * the same family.
 */
export function ConfirmDialog({
  open,
  onOpenChange,
  variant = 'destructive',
  icon,
  title,
  description,
  children,
  target,
  confirmLabel,
  cancelLabel,
  onConfirm,
  loading = false,
  disabled = false,
}: ConfirmDialogProps) {
  const { t } = useI18n();
  const cfg = VARIANT_CONFIG[variant];
  const resolvedIcon = icon ?? cfg.icon;

  // Belt-and-suspenders body pointer-events cleanup.
  //
  // Radix AlertDialog sets `pointer-events: none` on <body> while open
  // and restores it on close. But when a Dialog is opened from a
  // DropdownMenuItem, the menu's close cleanup races with the dialog's
  // open/close cleanup and sometimes leaves the body frozen. We can't
  // rely on a single "all done" moment, so whenever the dialog is not
  // open we forcibly clear the style on an interval for ~700ms. That
  // outlasts any Radix exit animation but is short enough to not leak.
  useEffect(() => {
    if (open) return;
    let attempts = 0;
    const id = window.setInterval(() => {
      document.body.style.pointerEvents = '';
      if (++attempts >= 7) window.clearInterval(id);
    }, 100);
    return () => window.clearInterval(id);
  }, [open]);

  // If this component unmounts while the dialog is mounted (e.g. parent
  // route change), make absolutely sure we release the body.
  useEffect(() => {
    return () => {
      document.body.style.pointerEvents = '';
    };
  }, []);

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent className="sm:max-w-md gap-0 p-0 overflow-hidden">
        <AlertDialogHeader className="px-5 pt-5 pb-3 border-b border-border">
          <AlertDialogTitle className="flex items-center gap-2 text-sm font-semibold">
            <span
              className={cn(
                'inline-flex items-center justify-center w-7 h-7 rounded-md shrink-0',
                cfg.iconBg,
                cfg.iconColor,
              )}
            >
              {resolvedIcon}
            </span>
            {title}
          </AlertDialogTitle>
          {description && (
            <AlertDialogDescription className="text-xs leading-relaxed mt-0.5 pl-9">
              {description}
            </AlertDialogDescription>
          )}
        </AlertDialogHeader>

        {(target || children) && (
          <div className="px-5 py-3 space-y-2">
            {target && (
              <div className="flex items-center gap-2 px-2.5 py-2 rounded-md bg-muted/40 border border-border">
                {target.label && (
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground shrink-0">
                    {target.label}
                  </span>
                )}
                <span className="text-xs font-medium truncate text-foreground">
                  {target.value}
                </span>
              </div>
            )}
            {children}
          </div>
        )}

        <AlertDialogFooter className="px-5 py-3 border-t border-border bg-muted/20">
          <AlertDialogCancel disabled={loading} className="h-8 text-xs">
            {cancelLabel ?? t('common.cancel')}
          </AlertDialogCancel>
          <AlertDialogAction
            onClick={(e) => {
              if (loading || disabled) {
                e.preventDefault();
                return;
              }
              onConfirm();
            }}
            disabled={loading || disabled}
            className={cn('h-8 text-xs min-w-[72px]', cfg.confirmClass)}
          >
            {loading ? (
              <Spinner size="sm" />
            ) : (
              confirmLabel ?? t('common.delete')
            )}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
