// Shared UI primitive class strings — the single source of truth for buttons,
// cards, inputs, and icon buttons across every view. These were first refined in
// KnowledgeView; promoting them here keeps every page visually aligned (one card
// surface, one primary button, one input) instead of each file redefining its own.
//
// Compose with cn() when a call site needs extra classes, e.g. cn(INPUT, "font-mono").

/** Elevated content card: the standard container for grouped settings/content. */
export const CARD =
  "professional-card p-4";

/** Text/number/select input. Pair with LABEL for a labelled field. */
export const INPUT =
  "w-full rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-elevated)] px-3 py-2 text-sm outline-none transition-colors focus:border-[var(--accent)] focus:shadow-[var(--shadow-focus)]";

/** Field caption above an input. */
export const LABEL = "mb-1.5 block text-xs font-medium text-[var(--text-muted)]";

/** Primary action — accent fill, white-on-accent text. */
export const BTN_PRIMARY =
  "inline-flex h-9 items-center justify-center gap-1.5 rounded-[var(--radius)] border border-[var(--border-strong)] bg-[var(--bg-elevated)] px-3.5 text-sm font-semibold text-[var(--text)] shadow-[0_1px_2px_rgba(15,23,42,0.08)] hover:border-[var(--text-muted)] hover:bg-[var(--surface)] disabled:opacity-50 transition-colors focus-ring";

/** Secondary action — bordered, neutral. */
export const BTN_GHOST =
  "inline-flex h-9 items-center justify-center gap-1.5 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-elevated)] px-3.5 text-sm text-[var(--text-muted)] shadow-[var(--shadow-sm)] hover:border-[var(--border-strong)] hover:bg-[var(--surface)] hover:text-[var(--text)] disabled:opacity-50 transition-colors focus-ring";

/** Destructive action — danger-tinted border/text, subtle fill on hover. */
export const BTN_DANGER =
  "inline-flex h-9 items-center justify-center gap-1.5 rounded-[var(--radius)] border border-[var(--danger)] bg-[var(--bg-elevated)] px-3.5 text-sm font-medium text-[var(--danger)] hover:bg-[var(--danger)]/10 disabled:opacity-50 transition-colors focus-ring";

/** Square icon button for toolbars/headers. */
export const ICON_BTN =
  "grid h-8 w-8 place-items-center rounded-[var(--radius)] text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] disabled:opacity-40 disabled:hover:bg-transparent transition-colors focus-ring";

/** The accent-tinted square badge that fronts a page title in headers. */
export const HEADER_BADGE =
  "grid h-7 w-7 flex-shrink-0 place-items-center rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] text-[var(--text-muted)]";

/** Base classes for a status pill; append a color set (border+bg+text). */
export const PILL_BASE =
  "rounded-full border px-2 py-0.5 text-[11px] font-medium whitespace-nowrap";
