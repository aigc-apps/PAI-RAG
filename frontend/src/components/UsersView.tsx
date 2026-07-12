import { Users } from "lucide-react";
import { UsersPanel } from "./UsersPanel";
import { PageHeader } from "./PageHeader";
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
      <PageHeader
        icon={Users}
        title={t("users.title")}
        onBack={onBack}
        backLabel={t("users.backToChat")}
      />
      <div className="min-h-0 flex-1 overflow-y-auto scrollbar-thin">
        <div className="mx-auto w-full max-w-4xl px-5 py-6">
          <UsersPanel />
        </div>
      </div>
    </div>
  );
}
