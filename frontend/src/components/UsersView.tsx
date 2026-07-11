import { ArrowLeft, Users } from "lucide-react";
import { UsersPanel } from "./UsersPanel";
import { ThemeToggle } from "./ThemeToggle";
import { useI18n } from "../i18n";

/**
 * User management as its own admin surface (mirrors KnowledgeView / SettingsView
 * shells). It lives outside Settings — Settings is agent-related configuration —
 * and is reached from the account menu.
 */
export function UsersView({ onBack }: { onBack: () => void }) {
  const { t } = useI18n();
  return (
    <div className="flex h-full flex-col bg-[var(--bg)] text-[var(--text)]">
      <div className="flex h-12 flex-shrink-0 items-center gap-2 border-b border-[var(--border)] px-3">
        <button
          type="button"
          aria-label={t("users.backToChat")}
          onClick={onBack}
          className="rounded-[var(--radius-sm)] p-1.5 text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
        >
          <ArrowLeft className="h-4 w-4" />
        </button>
        <span className="grid h-6 w-6 place-items-center rounded-[6px] bg-[var(--accent-soft)] text-[var(--accent)]">
          <Users className="h-3.5 w-3.5" />
        </span>
        <h1 className="text-sm font-semibold">{t("users.title")}</h1>
        <div className="flex-1" />
        <ThemeToggle />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto scrollbar-thin">
        <div className="mx-auto w-full max-w-4xl px-5 py-6">
          <UsersPanel />
        </div>
      </div>
    </div>
  );
}
