import { useState, type FormEvent, type ReactNode } from "react";
import { Loader2, ShieldCheck } from "lucide-react";
import { useAuthStore } from "../store/auth";

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
      <div className="w-full max-w-sm rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-6 shadow-xl">
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
      <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">{label}</span>
      <input
        type={type}
        value={value}
        placeholder={placeholder}
        autoFocus={autoFocus}
        autoComplete={autoComplete}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
      />
    </label>
  );
}

function SubmitButton({ busy, children }: { busy: boolean; children: ReactNode }) {
  return (
    <button
      type="submit"
      disabled={busy}
      className="mt-1 flex w-full items-center justify-center gap-2 rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-2 text-sm font-medium text-white disabled:opacity-60"
    >
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
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthShell title="Sign in" subtitle="Access your assistant workspace">
      <form onSubmit={submit}>
        <ErrorLine message={error} />
        <LabeledInput label="Email" type="email" value={email} onChange={setEmail}
          placeholder="you@example.com" autoFocus autoComplete="username" />
        <LabeledInput label="Password" type="password" value={password} onChange={setPassword}
          placeholder="••••••••" autoComplete="current-password" />
        <SubmitButton busy={busy}>Sign in</SubmitButton>
      </form>
    </AuthShell>
  );
}

// --------------------------------------------------------------------------- //
export function CreateAdminView() {
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
      setError("Password must be at least 8 characters");
      return;
    }
    if (password !== confirm) {
      setError("Passwords do not match");
      return;
    }
    setBusy(true);
    try {
      await createAdmin(email.trim(), password, token.trim() || undefined);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create the admin account");
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthShell title="Create admin account" subtitle="First-run setup — this becomes the owner">
      <form onSubmit={submit}>
        <ErrorLine message={error} />
        <LabeledInput label="Admin email" type="email" value={email} onChange={setEmail}
          placeholder="admin@example.com" autoFocus autoComplete="username" />
        <LabeledInput label="Password" type="password" value={password} onChange={setPassword}
          placeholder="At least 8 characters" autoComplete="new-password" />
        <LabeledInput label="Confirm password" type="password" value={confirm} onChange={setConfirm}
          placeholder="Repeat password" autoComplete="new-password" />
        <LabeledInput label="Setup token (only if configured)" type="password" value={token}
          onChange={setToken} placeholder="Optional" />
        <SubmitButton busy={busy}>Create account</SubmitButton>
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
  const acceptInvite = useAuthStore((s) => s.acceptInvite);
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    if (password.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }
    if (password !== confirm) {
      setError("Passwords do not match");
      return;
    }
    setBusy(true);
    try {
      await acceptInvite(token, password);
      onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : "This invite is invalid or has expired");
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthShell title="Set your password" subtitle="Finish activating your invited account">
      <form onSubmit={submit}>
        <ErrorLine message={error} />
        <LabeledInput label="New password" type="password" value={password} onChange={setPassword}
          placeholder="At least 8 characters" autoFocus autoComplete="new-password" />
        <LabeledInput label="Confirm password" type="password" value={confirm} onChange={setConfirm}
          placeholder="Repeat password" autoComplete="new-password" />
        <SubmitButton busy={busy}>Activate account</SubmitButton>
      </form>
    </AuthShell>
  );
}
