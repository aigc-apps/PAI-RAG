// Shared UI primitive class strings — the single source of truth for buttons,
// cards, inputs, and icon buttons across every view. These were first refined in
// KnowledgeView; promoting them here keeps every page visually aligned (one card
// surface, one primary button, one input) instead of each file redefining its own.
//
// Compose with cn() when a call site needs extra classes, e.g. cn(INPUT, "font-mono").

/** Elevated content card: the standard container for grouped settings/content. */
export const CARD =
  "rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-elevated)] p-4";

/** Text/number/select input. Pair with LABEL for a labelled field. */
export const INPUT =
  "w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm outline-none focus:border-[var(--accent)]";

/** Field caption above an input. */
export const LABEL = "mb-1.5 block text-xs font-medium text-[var(--text-muted)]";

/** Primary action — accent fill, white-on-accent text. */
export const BTN_PRIMARY =
  "inline-flex items-center justify-center gap-1.5 rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-2 text-sm font-medium text-[var(--accent-fg)] hover:bg-[var(--accent-hover)] disabled:opacity-50 transition-colors";

/** Secondary action — bordered, neutral. */
export const BTN_GHOST =
  "inline-flex items-center justify-center gap-1.5 rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-2 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] disabled:opacity-50 transition-colors";

/** Destructive action — danger-tinted border/text, subtle fill on hover. */
export const BTN_DANGER =
  "inline-flex items-center justify-center gap-1.5 rounded-[var(--radius-sm)] border border-[var(--danger)] px-3 py-2 text-sm font-medium text-[var(--danger)] hover:bg-[var(--danger)]/10 disabled:opacity-50 transition-colors";

/** Square icon button for toolbars/headers. */
export const ICON_BTN =
  "grid h-8 w-8 place-items-center rounded-[var(--radius-sm)] text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] disabled:opacity-40 disabled:hover:bg-transparent transition-colors";

/** The accent-tinted square badge that fronts a page title in headers. */
export const HEADER_BADGE =
  "grid h-6 w-6 flex-shrink-0 place-items-center rounded-[var(--radius-sm)] bg-[var(--accent-soft)] text-[var(--accent)]";

/** Base classes for a status pill; append a color set (border+bg+text). */
export const PILL_BASE =
  "rounded-full border px-2 py-0.5 text-[11px] font-medium whitespace-nowrap";
