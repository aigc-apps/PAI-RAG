import { useEffect, useState, type FormEvent } from "react";
import { ChevronsUpDown, KeyRound, Languages, LogOut, Settings2, ShieldCheck, Users } from "lucide-react";
import { toast } from "sonner";
import { useAuthStore } from "../store/auth";
import { changePassword } from "../api/auth";
import { getAliyunStatus } from "../api/agentConfig";
import { useAliyunDialog } from "../store/aliyunDialog";
import { useI18n } from "../i18n";

function initialOf(email: string | null | undefined): string {
  return (email?.trim()?.[0] ?? "?").toUpperCase();
}

function ChangePasswordDialog({ onClose }: { onClose: () => void }) {
  const { t } = useI18n();
  const [oldPassword, setOld] = useState("");
  const [newPassword, setNew] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    if (newPassword.length < 8) {
      setError(t("userMenu.pwTooShort"));
      return;
    }
    if (newPassword !== confirm) {
      setError(t("userMenu.pwMismatch"));
      return;
    }
    setBusy(true);
    try {
      await changePassword(oldPassword, newPassword);
      toast.success(t("userMenu.pwUpdated"));
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : t("userMenu.pwChangeFailed"));
    } finally {
      setBusy(false);
    }
  };

  const inputCls =
    "mb-3 w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm outline-none focus:border-[var(--accent)]";

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/30 p-4">
      <div className="w-full max-w-sm rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-5 shadow-xl">
        <div className="mb-4 text-sm font-semibold">{t("userMenu.changePassword")}</div>
        <form onSubmit={submit}>
          {error && (
            <div className="mb-3 rounded-[var(--radius-sm)] bg-[var(--danger)]/10 px-3 py-2 text-xs text-[var(--danger)]">
              {error}
            </div>
          )}
          <input type="password" value={oldPassword} onChange={(e) => setOld(e.target.value)}
            placeholder={t("userMenu.pwCurrent")} autoComplete="current-password" className={inputCls} />
          <input type="password" value={newPassword} onChange={(e) => setNew(e.target.value)}
            placeholder={t("userMenu.pwNew")} autoComplete="new-password" className={inputCls} />
          <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)}
            placeholder={t("userMenu.pwConfirm")} autoComplete="new-password" className={inputCls} />
          <div className="mt-1 flex justify-end gap-2">
            <button type="button" onClick={onClose}
              className="rounded-[var(--radius-sm)] px-3 py-1.5 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)]">
              {t("common.cancel")}
            </button>
            <button type="submit" disabled={busy}
              className="rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-1.5 text-sm font-medium text-white disabled:opacity-60">
              {t("common.save")}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export function UserMenu({
  onOpenSettings,
  onOpenUsers,
}: {
  onOpenSettings?: () => void;
  onOpenUsers?: () => void;
}) {
  const { t, lang, setLang } = useI18n();
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const [open, setOpen] = useState(false);
  const showAliyunDialog = useAliyunDialog((s) => s.show);
  const [showChangePw, setShowChangePw] = useState(false);
  const [aliyunConfigured, setAliyunConfigured] = useState(false);

  // Probe once whether cross-account authorization is enabled on this
  // deployment; the menu item is only actionable when it is.
  useEffect(() => {
    let alive = true;
    getAliyunStatus()
      .then((s) => alive && setAliyunConfigured(Boolean(s.configured)))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);

  const itemCls =
    "flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-[var(--text)] hover:bg-[var(--surface-2)] disabled:opacity-50 disabled:hover:bg-transparent";

  return (
    <>
      <div className="relative">
        <button
          type="button"
          aria-label={t("userMenu.accountMenu")}
          onClick={() => setOpen((o) => !o)}
          className="flex w-full items-center gap-2 rounded-[var(--radius-sm)] px-2 py-1.5 text-left hover:bg-[var(--surface-2)]"
        >
          <span className="grid h-7 w-7 flex-shrink-0 place-items-center rounded-full bg-[var(--accent)]/15 text-xs font-semibold text-[var(--accent)]">
            {initialOf(user?.email)}
          </span>
          <span className="min-w-0 flex-1 leading-tight">
            <span className="block truncate text-sm font-medium text-[var(--text)]">
              {user?.email ?? t("userMenu.account")}
            </span>
            {user?.role && (
              <span className="block truncate text-xs capitalize text-[var(--text-muted)]">
                {user.role}
              </span>
            )}
          </span>
          <ChevronsUpDown className="h-4 w-4 flex-shrink-0 text-[var(--text-muted)]" />
        </button>

        {open && (
          <>
            {/* click-away backdrop */}
            <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
            {/* Opens upward + left-aligned — the menu lives at the bottom-left of the sidebar. */}
            <div className="absolute left-0 bottom-full z-50 mb-2 w-56 overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] shadow-xl">
              <div className="border-b border-[var(--border)] px-3 py-2">
                <div className="truncate text-sm font-medium">{user?.email ?? t("userMenu.account")}</div>
                <div className="text-xs capitalize text-[var(--text-muted)]">{user?.role}</div>
              </div>
              {onOpenSettings && (
                <button
                  type="button"
                  className={itemCls}
                  onClick={() => {
                    setOpen(false);
                    onOpenSettings();
                  }}
                >
                  <Settings2 className="h-4 w-4 text-[var(--text-muted)]" />
                  {t("userMenu.settings")}
                </button>
              )}
              {onOpenUsers && (
                <button
                  type="button"
                  className={itemCls}
                  onClick={() => {
                    setOpen(false);
                    onOpenUsers();
                  }}
                >
                  <Users className="h-4 w-4 text-[var(--text-muted)]" />
                  {t("userMenu.users")}
                </button>
              )}
              <button
                type="button"
                className={itemCls}
                disabled={!aliyunConfigured}
                title={aliyunConfigured ? undefined : t("userMenu.aliyunDisabled")}
                onClick={() => {
                  setOpen(false);
                  showAliyunDialog();
                }}
              >
                <ShieldCheck className="h-4 w-4 text-[var(--text-muted)]" />
                {t("userMenu.aliyunAuth")}
              </button>
              <div className="flex items-center gap-2 border-t border-[var(--border)] px-3 py-2 text-sm text-[var(--text)]">
                <Languages className="h-4 w-4 flex-shrink-0 text-[var(--text-muted)]" />
                <span className="flex-1">{t("lang.label")}</span>
                <div className="flex overflow-hidden rounded-[var(--radius-sm)] border border-[var(--border)]">
                  {(["zh", "en"] as const).map((code) => (
                    <button
                      key={code}
                      type="button"
                      onClick={() => setLang(code)}
                      className={
                        "px-2 py-0.5 text-xs transition-colors " +
                        (lang === code
                          ? "bg-[var(--accent)] text-white"
                          : "text-[var(--text-muted)] hover:bg-[var(--surface-2)]")
                      }
                    >
                      {t(code === "zh" ? "lang.zh" : "lang.en")}
                    </button>
                  ))}
                </div>
              </div>
              <button
                type="button"
                className={itemCls}
                onClick={() => {
                  setOpen(false);
                  setShowChangePw(true);
                }}
              >
                <KeyRound className="h-4 w-4 text-[var(--text-muted)]" />
                {t("userMenu.changePassword")}
              </button>
              <button
                type="button"
                className={itemCls}
                onClick={() => {
                  setOpen(false);
                  void logout();
                }}
              >
                <LogOut className="h-4 w-4 text-[var(--text-muted)]" />
                {t("userMenu.signOut")}
              </button>
            </div>
          </>
        )}
      </div>

      {showChangePw && <ChangePasswordDialog onClose={() => setShowChangePw(false)} />}
    </>
  );
}
