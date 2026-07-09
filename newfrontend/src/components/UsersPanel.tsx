import { useEffect, useState } from "react";
import { Copy, UserPlus } from "lucide-react";
import { toast } from "sonner";
import {
  inviteUser,
  listUsers,
  setUserStatus,
  type AuthUser,
  type InviteResult,
  type Role,
} from "../api/auth";
import { useAuthStore } from "../store/auth";
import { cn } from "../lib/cn";
import { copyText } from "../lib/clipboard";

export function UsersPanel() {
  const me = useAuthStore((s) => s.user);
  const [users, setUsers] = useState<AuthUser[]>([]);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<Role>("user");
  const [invite, setInvite] = useState<InviteResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const refresh = async () => {
    try {
      setUsers(await listUsers());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load users");
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const submitInvite = async () => {
    setError("");
    if (!email.trim() || !email.includes("@")) {
      setError("Enter a valid email address");
      return;
    }
    setBusy(true);
    try {
      const result = await inviteUser(email.trim(), role);
      setInvite(result);
      setEmail("");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create the invite");
    } finally {
      setBusy(false);
    }
  };

  const copyLink = async (url: string) => {
    if (await copyText(url)) {
      toast.success("Invite link copied");
    } else {
      toast.error("Could not copy — select and copy manually");
    }
  };

  const toggleStatus = async (u: AuthUser) => {
    const next = u.status === "disabled" ? "active" : "disabled";
    try {
      await setUserStatus(u.id, next);
      await refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not update user");
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="mb-1 text-sm font-semibold">Users</h2>
        <p className="text-xs text-[var(--text-muted)]">
          Invite teammates by email. There is no mail server — copy the invite link
          and send it yourself. The invitee sets their own password to activate.
        </p>
      </div>

      {/* Invite form */}
      <div className="rounded-[var(--radius-sm)] border border-[var(--border)] p-3">
        <div className="flex flex-wrap items-end gap-2">
          <label className="flex-1 text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Email</span>
            <input
              type="email"
              value={email}
              placeholder="teammate@example.com"
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
            />
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Role</span>
            <select
              value={role}
              onChange={(e) => setRole(e.target.value as Role)}
              className="rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
            >
              <option value="user">User</option>
              <option value="admin">Admin</option>
            </select>
          </label>
          <button
            type="button"
            disabled={busy}
            onClick={submitInvite}
            className="inline-flex items-center gap-2 rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-2 text-sm font-medium text-white disabled:opacity-60"
          >
            <UserPlus className="h-4 w-4" />
            Invite
          </button>
        </div>
        {error && (
          <div className="mt-2 rounded-[var(--radius-sm)] bg-[var(--danger)]/10 px-3 py-2 text-xs text-[var(--danger)]">
            {error}
          </div>
        )}
        {invite && (
          <div className="mt-3 rounded-[var(--radius-sm)] border border-[var(--success)]/40 bg-[var(--success)]/10 p-3">
            <div className="mb-1 text-xs font-medium text-[var(--success)]">
              Invite created for {invite.user.email} — copy this link:
            </div>
            <div className="flex items-center gap-2">
              <code className="flex-1 break-all rounded-[var(--radius-sm)] bg-[var(--bg)] px-2 py-1.5 font-mono text-xs text-[var(--text-muted)]">
                {invite.invite_url}
              </code>
              <button
                type="button"
                aria-label="Copy invite link"
                onClick={() => copyLink(invite.invite_url)}
                className="shrink-0 rounded-[var(--radius-sm)] border border-[var(--border)] p-2 hover:bg-[var(--surface-2)]"
              >
                <Copy className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-1 text-[11px] text-[var(--text-faint)]">
              Expires {new Date(invite.expires_at).toLocaleString()}
            </div>
          </div>
        )}
      </div>

      {/* User list */}
      <div className="overflow-hidden rounded-[var(--radius-sm)] border border-[var(--border)]">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--border)] text-left text-xs text-[var(--text-muted)]">
              <th className="px-3 py-2 font-medium">Email</th>
              <th className="px-3 py-2 font-medium">Role</th>
              <th className="px-3 py-2 font-medium">Status</th>
              <th className="px-3 py-2" />
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id} className="border-b border-[var(--border)] last:border-0">
                <td className="px-3 py-2">
                  {u.email}
                  {u.id === me?.id && (
                    <span className="ml-1.5 text-[11px] text-[var(--text-faint)]">(you)</span>
                  )}
                </td>
                <td className="px-3 py-2 capitalize text-[var(--text-muted)]">{u.role}</td>
                <td className="px-3 py-2">
                  <span
                    className={cn(
                      "capitalize",
                      u.status === "active" && "text-[var(--success)]",
                      u.status === "invited" && "text-[var(--warning)]",
                      u.status === "disabled" && "text-[var(--text-faint)]"
                    )}
                  >
                    {u.status}
                  </span>
                </td>
                <td className="px-3 py-2 text-right">
                  {u.id !== me?.id && u.status !== "invited" && (
                    <button
                      type="button"
                      onClick={() => toggleStatus(u)}
                      className="rounded-[var(--radius-sm)] border border-[var(--border)] px-2 py-1 text-xs hover:bg-[var(--surface-2)]"
                    >
                      {u.status === "disabled" ? "Enable" : "Disable"}
                    </button>
                  )}
                </td>
              </tr>
            ))}
            {users.length === 0 && (
              <tr>
                <td colSpan={4} className="px-3 py-4 text-center text-xs text-[var(--text-faint)]">
                  No users yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
