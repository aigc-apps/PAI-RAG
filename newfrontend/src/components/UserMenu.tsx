import { useEffect, useState, type FormEvent } from "react";
import { ChevronsUpDown, KeyRound, LogOut, Settings2, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import { useAuthStore } from "../store/auth";
import { changePassword } from "../api/auth";
import { getAliyunStatus } from "../api/agentConfig";
import { useAliyunDialog } from "../store/aliyunDialog";

function initialOf(email: string | null | undefined): string {
  return (email?.trim()?.[0] ?? "?").toUpperCase();
}

function ChangePasswordDialog({ onClose }: { onClose: () => void }) {
  const [oldPassword, setOld] = useState("");
  const [newPassword, setNew] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    if (newPassword.length < 8) {
      setError("New password must be at least 8 characters");
      return;
    }
    if (newPassword !== confirm) {
      setError("Passwords do not match");
      return;
    }
    setBusy(true);
    try {
      await changePassword(oldPassword, newPassword);
      toast.success("Password updated");
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not change password");
    } finally {
      setBusy(false);
    }
  };

  const inputCls =
    "mb-3 w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm outline-none focus:border-[var(--accent)]";

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/30 p-4">
      <div className="w-full max-w-sm rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-5 shadow-xl">
        <div className="mb-4 text-sm font-semibold">Change password</div>
        <form onSubmit={submit}>
          {error && (
            <div className="mb-3 rounded-[var(--radius-sm)] bg-[var(--danger)]/10 px-3 py-2 text-xs text-[var(--danger)]">
              {error}
            </div>
          )}
          <input type="password" value={oldPassword} onChange={(e) => setOld(e.target.value)}
            placeholder="Current password" autoComplete="current-password" className={inputCls} />
          <input type="password" value={newPassword} onChange={(e) => setNew(e.target.value)}
            placeholder="New password" autoComplete="new-password" className={inputCls} />
          <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)}
            placeholder="Confirm new password" autoComplete="new-password" className={inputCls} />
          <div className="mt-1 flex justify-end gap-2">
            <button type="button" onClick={onClose}
              className="rounded-[var(--radius-sm)] px-3 py-1.5 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)]">
              Cancel
            </button>
            <button type="submit" disabled={busy}
              className="rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-1.5 text-sm font-medium text-white disabled:opacity-60">
              Save
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export function UserMenu({ onOpenSettings }: { onOpenSettings?: () => void }) {
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
          aria-label="Account menu"
          onClick={() => setOpen((o) => !o)}
          className="flex w-full items-center gap-2 rounded-[var(--radius-sm)] px-2 py-1.5 text-left hover:bg-[var(--surface-2)]"
        >
          <span className="grid h-7 w-7 flex-shrink-0 place-items-center rounded-full bg-[var(--accent)]/15 text-xs font-semibold text-[var(--accent)]">
            {initialOf(user?.email)}
          </span>
          <span className="min-w-0 flex-1 leading-tight">
            <span className="block truncate text-sm font-medium text-[var(--text)]">
              {user?.email ?? "Account"}
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
                <div className="truncate text-sm font-medium">{user?.email ?? "Account"}</div>
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
                  Settings
                </button>
              )}
              <button
                type="button"
                className={itemCls}
                disabled={!aliyunConfigured}
                title={aliyunConfigured ? undefined : "Not enabled on this deployment"}
                onClick={() => {
                  setOpen(false);
                  showAliyunDialog();
                }}
              >
                <ShieldCheck className="h-4 w-4 text-[var(--text-muted)]" />
                Aliyun authorization
              </button>
              <button
                type="button"
                className={itemCls}
                onClick={() => {
                  setOpen(false);
                  setShowChangePw(true);
                }}
              >
                <KeyRound className="h-4 w-4 text-[var(--text-muted)]" />
                Change password
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
                Sign out
              </button>
            </div>
          </>
        )}
      </div>

      {showChangePw && <ChangePasswordDialog onClose={() => setShowChangePw(false)} />}
    </>
  );
}
