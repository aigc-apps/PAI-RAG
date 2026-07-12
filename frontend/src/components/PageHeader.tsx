import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { ArrowLeft } from "lucide-react";
import { ICON_BTN, HEADER_BADGE } from "../lib/ui";

/**
 * The shared top bar for full-page admin surfaces (Settings / Knowledge / Users).
 * One structure everywhere: an optional back button, an accent-tinted icon badge,
 * the page title (or a breadcrumb node), and a right-aligned actions slot.
 *
 * ChatView keeps its own toolbar — its centered title + agent selector layout is
 * deliberately different from these landing/detail pages.
 */
export function PageHeader({
  icon: Icon,
  title,
  onBack,
  backLabel = "Back",
  actions,
}: {
  icon: LucideIcon;
  /** A plain string title, or a breadcrumb/rich node. */
  title: ReactNode;
  onBack?: () => void;
  backLabel?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex h-12 flex-shrink-0 items-center gap-2 border-b border-[var(--border)] px-3">
      {onBack && (
        <button type="button" aria-label={backLabel} onClick={onBack} className={ICON_BTN}>
          <ArrowLeft className="h-4 w-4" />
        </button>
      )}
      <span className={HEADER_BADGE}>
        <Icon className="h-3.5 w-3.5" />
      </span>
      {typeof title === "string" ? (
        <h1 className="truncate text-sm font-semibold text-[var(--text)]">{title}</h1>
      ) : (
        <div className="min-w-0 truncate text-[13px] text-[var(--text-muted)]">{title}</div>
      )}
      <div className="flex-1" />
      {actions}
    </div>
  );
}
