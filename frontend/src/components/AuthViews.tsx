import { useState, type FormEvent, type ReactNode } from "react";
import { Loader2, ShieldCheck } from "lucide-react";
import { useAuthStore } from "../store/auth";
import { cn } from "../lib/cn";
import { INPUT, LABEL, BTN_PRIMARY } from "../lib/ui";
import { useI18n } from "../i18n";

// A centered card shell shared by the three unauthenticated screens.
function AuthShell({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
}) {
  return (
    <div className="grid h-full place-items-center bg-[var(--bg)] p-4">
      <div className="w-full max-w-sm rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-elevated)] p-6 shadow-xl">
        <div className="mb-5 flex items-center gap-2">
          <div className="grid h-8 w-8 place-items-center rounded-[var(--radius-sm)] bg-[var(--accent)]/10 text-[var(--accent)]">
            <ShieldCheck className="h-4 w-4" />
          </div>
          <div>
            <div className="text-sm font-semibold">{title}</div>
            <div className="text-xs text-[var(--text-muted)]">{subtitle}</div>
          </div>
        </div>
        {children}
      </div>
    </div>
  );
}

function LabeledInput({
  label,
  type,
  value,
  onChange,
  placeholder,
  autoFocus,
  autoComplete,
}: {
  label: string;
  type: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  autoFocus?: boolean;
  autoComplete?: string;
}) {
  return (
    <label className="mb-3 block text-sm">
      <span className={LABEL}>{label}</span>
      <input
        type={type}
        value={value}
        placeholder={placeholder}
        autoFocus={autoFocus}
        autoComplete={autoComplete}
        onChange={(e) => onChange(e.target.value)}
        className={INPUT}
      />
    </label>
  );
}

function SubmitButton({ busy, children }: { busy: boolean; children: ReactNode }) {
  return (
    <button type="submit" disabled={busy} className={cn(BTN_PRIMARY, "mt-1 w-full")}>
      {busy && <Loader2 className="h-4 w-4 animate-spin" />}
      {children}
    </button>
  );
}

function ErrorLine({ message }: { message: string }) {
  if (!message) return null;
  return (
    <div className="mb-3 rounded-[var(--radius-sm)] bg-[var(--danger)]/10 px-3 py-2 text-xs text-[var(--danger)]">
      {message}
    </div>
  );
}

// --------------------------------------------------------------------------- //
export function LoginView() {
  const { t } = useI18n();
  const login = useAuthStore((s) => s.login);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await login(email.trim(), password);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("auth.loginFailed"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthShell title={t("auth.signIn")} subtitle={t("auth.signInSubtitle")}>
      <form onSubmit={submit}>
        <ErrorLine message={error} />
        <LabeledInput label={t("auth.email")} type="email" value={email} onChange={setEmail}
          placeholder="you@example.com" autoFocus autoComplete="username" />
        <LabeledInput label={t("auth.password")} type="password" value={password} onChange={setPassword}
          placeholder="••••••••" autoComplete="current-password" />
        <SubmitButton busy={busy}>{t("auth.signIn")}</SubmitButton>
      </form>
    </AuthShell>
  );
}

// --------------------------------------------------------------------------- //
export function CreateAdminView() {
  const { t } = useI18n();
  const createAdmin = useAuthStore((s) => s.createAdmin);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    if (password.length < 8) {
      setError(t("auth.pwTooShort"));
      return;
    }
    if (password !== confirm) {
      setError(t("auth.pwMismatch"));
      return;
    }
    setBusy(true);
    try {
      await createAdmin(email.trim(), password, token.trim() || undefined);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("auth.createAdminFailed"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthShell title={t("auth.createAdmin")} subtitle={t("auth.createAdminSubtitle")}>
      <form onSubmit={submit}>
        <ErrorLine message={error} />
        <LabeledInput label={t("auth.adminEmail")} type="email" value={email} onChange={setEmail}
          placeholder="admin@example.com" autoFocus autoComplete="username" />
        <LabeledInput label={t("auth.password")} type="password" value={password} onChange={setPassword}
          placeholder={t("auth.atLeast8")} autoComplete="new-password" />
        <LabeledInput label={t("auth.confirmPassword")} type="password" value={confirm} onChange={setConfirm}
          placeholder={t("auth.repeatPassword")} autoComplete="new-password" />
        <LabeledInput label={t("auth.setupToken")} type="password" value={token}
          onChange={setToken} placeholder={t("auth.optional")} />
        <SubmitButton busy={busy}>{t("auth.createAccount")}</SubmitButton>
      </form>
    </AuthShell>
  );
}

// --------------------------------------------------------------------------- //
export function AcceptInviteView({
  token,
  onDone,
}: {
  token: string;
  onDone: () => void;
}) {
  const { t } = useI18n();
  const acceptInvite = useAuthStore((s) => s.acceptInvite);
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    if (password.length < 8) {
      setError(t("auth.pwTooShort"));
      return;
    }
    if (password !== confirm) {
      setError(t("auth.pwMismatch"));
      return;
    }
    setBusy(true);
    try {
      await acceptInvite(token, password);
      onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : t("auth.inviteInvalid"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthShell title={t("auth.setPassword")} subtitle={t("auth.setPasswordSubtitle")}>
      <form onSubmit={submit}>
        <ErrorLine message={error} />
        <LabeledInput label={t("auth.newPassword")} type="password" value={password} onChange={setPassword}
          placeholder={t("auth.atLeast8")} autoFocus autoComplete="new-password" />
        <LabeledInput label={t("auth.confirmPassword")} type="password" value={confirm} onChange={setConfirm}
          placeholder={t("auth.repeatPassword")} autoComplete="new-password" />
        <SubmitButton busy={busy}>{t("auth.activateAccount")}</SubmitButton>
      </form>
    </AuthShell>
  );
}
