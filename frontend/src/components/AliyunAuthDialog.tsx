import { useEffect, useState } from "react";
import {
  ArrowRight,
  CircleAlert,
  CircleCheck,
  ExternalLink,
  ShieldCheck,
  X,
} from "lucide-react";
import { cn } from "../lib/cn";
import { useAliyunDialog } from "../store/aliyunDialog";
import { useComposer } from "../store/composer";
import {
  authorizeAliyun,
  deauthorizeAliyun,
  getAliyunStatus,
  type AliyunStatus,
} from "../api/agentConfig";
import { useI18n } from "../i18n";

/** Cross-account Aliyun PAI authorization. Identity is the session user; the
 * dialog only ever sends the RoleArn. Reachable from the avatar menu (any user)
 * and from Settings → Tools (admin). */
export function AliyunAuthDialog({ onClose }: { onClose: () => void }) {
  const { t } = useI18n();
  const [status, setStatus] = useState<AliyunStatus | null>(null);
  const [roleArn, setRoleArn] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [verdict, setVerdict] = useState<string>("");
  // True when opened from a paused agent turn: offer a "继续" button on success.
  const resumeAfter = useAliyunDialog((s) => s.resumeAfter);
  const [authorized, setAuthorized] = useState(false);

  const refresh = async () => {
    try {
      setStatus(await getAliyunStatus());
    } catch (err) {
      setError(err instanceof Error ? err.message : t("aliyun.loadStatusFailed"));
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const submit = async () => {
    setError("");
    setVerdict("");
    if (!roleArn.trim()) {
      setError(t("aliyun.pasteFirst"));
      return;
    }
    setBusy(true);
    try {
      const result = await authorizeAliyun({ role_arn: roleArn.trim() });
      if (result.ok) {
        const v = result.verdict;
        const withServices = (v.regions ?? []).filter(
          (r) => r.ok && (r.pai_total ?? 0) > 0
        );
        const detail = withServices.length
          ? t("aliyun.detailServices", {
              list: withServices.map((r) => `${r.region} (${r.pai_total})`).join(", "),
            })
          : v.pai_total != null
            ? t("aliyun.detailCount", { count: v.pai_total })
            : t("aliyun.detailNone");
        setVerdict(t("aliyun.authorizedDetail", { account: v.account_id ?? "?", detail }));
        setRoleArn("");
        setAuthorized(true);
        await refresh();
      } else {
        setError(
          result.verdict.error_message ||
            result.verdict.error_code ||
            "Authorization failed — check the role and try again"
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t("aliyun.requestFailed"));
    } finally {
      setBusy(false);
    }
  };

  const deauthorize = async () => {
    setError("");
    setVerdict("");
    setBusy(true);
    try {
      await deauthorizeAliyun();
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : t("aliyun.revokeFailed"));
    } finally {
      setBusy(false);
    }
  };

  const configured = status?.configured ?? false;

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/30 p-4">
      <div className="max-h-[92vh] w-full max-w-2xl overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg)] shadow-xl">
        <div className="flex h-11 items-center border-b border-[var(--border)] px-4">
          <ShieldCheck className="mr-2 h-4 w-4 text-[var(--text-muted)]" />
          <div className="text-sm font-semibold">{t("aliyun.title")}</div>
          <button
            type="button"
            aria-label={t("aliyun.close")}
            onClick={onClose}
            className="ml-auto rounded-[var(--radius-sm)] p-1 text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="max-h-[calc(92vh-96px)] space-y-4 overflow-y-auto p-4">
          <p className="text-sm leading-6 text-[var(--text-muted)]">{t("aliyun.intro")}</p>

          {!configured && (
            <div className="flex gap-2 rounded-[var(--radius-sm)] bg-[var(--warning)]/10 px-3 py-2 text-xs text-[var(--warning)]">
              <CircleAlert className="h-3.5 w-3.5 shrink-0" />
              {t("aliyun.notConfigured")}
            </div>
          )}

          {status?.bound ? (
            <div className="space-y-3 rounded-[var(--radius-sm)] border border-[var(--success)]/40 bg-[var(--success)]/10 p-3">
              <div className="flex items-center gap-2 text-sm font-medium text-[var(--success)]">
                <CircleCheck className="h-4 w-4" />
                {t("aliyun.authorized")}
              </div>
              <dl className="grid gap-1 text-xs text-[var(--text-muted)]">
                <div className="flex gap-2">
                  <dt className="w-28 shrink-0 text-[var(--text-faint)]">{t("aliyun.account")}</dt>
                  <dd className="font-mono">{status.assumed_account_id ?? "—"}</dd>
                </div>
                <div className="flex gap-2">
                  <dt className="w-28 shrink-0 text-[var(--text-faint)]">{t("aliyun.roleArn")}</dt>
                  <dd className="break-all font-mono">{status.role_arn ?? "—"}</dd>
                </div>
                <div className="flex gap-2">
                  <dt className="w-28 shrink-0 text-[var(--text-faint)]">{t("aliyun.verified")}</dt>
                  <dd>{status.verified_at ? new Date(status.verified_at).toLocaleString() : "—"}</dd>
                </div>
                <div className="flex gap-2">
                  <dt className="w-28 shrink-0 text-[var(--text-faint)]">{t("aliyun.regions")}</dt>
                  <dd>
                    {(() => {
                      const totals = status.region_totals ?? {};
                      const svc = status.service_regions ?? [];
                      const reachable = status.regions ?? [];
                      if (svc.length) {
                        return svc
                          .map((r) => `${r}${totals[r] != null ? ` (${totals[r]})` : ""}`)
                          .join(", ");
                      }
                      if (reachable.length) {
                        return `${reachable.join(", ")} — ${t("aliyun.noServicesYet")}`;
                      }
                      return "—";
                    })()}
                  </dd>
                </div>
              </dl>
              <button
                type="button"
                disabled={busy}
                onClick={deauthorize}
                className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-1.5 text-xs hover:bg-[var(--surface-2)] disabled:opacity-60"
              >
                {t("aliyun.revoke")}
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="rounded-[var(--radius-sm)] border border-[var(--border)] p-3">
                <div className="mb-2 text-xs font-semibold text-[var(--text-muted)]">
                  {t("aliyun.step1")}
                </div>
                <a
                  href={status?.ros_url ?? undefined}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-disabled={!status?.ros_url}
                  className={cn(
                    "inline-flex items-center gap-2 rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-1.5 text-sm font-medium text-white",
                    !status?.ros_url && "pointer-events-none opacity-50"
                  )}
                >
                  <ExternalLink className="h-4 w-4" />
                  {t("aliyun.authorizeOnAliyun")}
                </a>
                {status?.external_id && (
                  <div className="mt-2 flex gap-2 text-xs text-[var(--text-faint)]">
                    <span className="shrink-0">ExternalId</span>
                    <code className="break-all font-mono text-[var(--text-muted)]">
                      {status.external_id}
                    </code>
                  </div>
                )}
              </div>

              <div className="rounded-[var(--radius-sm)] border border-[var(--border)] p-3">
                <div className="mb-2 text-xs font-semibold text-[var(--text-muted)]">
                  {t("aliyun.step2")}
                </div>
                <input
                  aria-label={t("aliyun.roleArnAria")}
                  value={roleArn}
                  placeholder="acs:ram::<account>:role/pai-authz-crossaccount-role"
                  onChange={(event) => setRoleArn(event.target.value)}
                  className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 font-mono text-xs"
                />
                <button
                  type="button"
                  disabled={busy || !configured}
                  onClick={submit}
                  className="mt-2 rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-1.5 text-sm font-medium text-white disabled:opacity-60"
                >
                  {busy ? t("aliyun.verifying") : t("aliyun.verifyAuthorize")}
                </button>
              </div>
            </div>
          )}

          {verdict && (
            <div className="flex gap-2 rounded-[var(--radius-sm)] bg-[var(--success)]/10 px-3 py-2 text-xs text-[var(--success)]">
              <CircleCheck className="h-3.5 w-3.5 shrink-0" />
              {verdict}
            </div>
          )}
          {authorized && resumeAfter && (
            <button
              type="button"
              onClick={() => {
                useComposer.getState().submit?.(t("aliyun.resumeMessage"));
                onClose();
              }}
              className="inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-1.5 text-sm font-medium text-white hover:opacity-90"
            >
              {t("aliyun.resumeButton")}
              <ArrowRight className="h-4 w-4" />
            </button>
          )}
          {error && (
            <div className="rounded-[var(--radius-sm)] bg-[var(--danger)]/10 px-3 py-2 text-xs text-[var(--danger)]">
              {error}
            </div>
          )}
        </div>
        <div className="flex justify-end gap-2 border-t border-[var(--border)] px-4 py-3">
          <button
            type="button"
            onClick={onClose}
            className="rounded-[var(--radius-sm)] px-3 py-1.5 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
          >
            {t("common.close")}
          </button>
        </div>
      </div>
    </div>
  );
}
