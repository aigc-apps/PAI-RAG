import { useState } from "react";
import { ArrowRight, CircleAlert, CircleCheck, History, Loader2, ShieldCheck } from "lucide-react";
import type { ToolUse } from "../types";
import { cn } from "../lib/cn";
import { useAliyunDialog } from "../store/aliyunDialog";
import { useComposer } from "../store/composer";
import { verifyAliyun } from "../api/agentConfig";
import { useI18n, translate, useI18nStore } from "../i18n";

/** Resume the paused agent turn by sending a continue message on the user's behalf. */
function resumeAgent() {
  const text = translate(useI18nStore.getState().lang, "aliyun.resumeMessage");
  useComposer.getState().submit?.(text);
}

/**
 * Inline authorization prompt rendered in place of a `<ToolCall>` when the shell
 * tool reports an aliyun credential failure. Lazy by design — it only appears
 * after an actual `aliyun` CLI call failed. `bound === false` means the user
 * never authorized (offer "去授权"); `bound === true` means the binding exists
 * but the creds are broken/expired (offer an in-place re-verify, plus a path to
 * re-authorize). The real ROS flow lives in {@link AliyunAuthDialog}.
 *
 * `historical` renders the persisted card as a read-only record: on reload, a
 * card that isn't the latest assistant message has already been handled (a later
 * turn exists), so it drops the action buttons and shows a muted "已处理" badge.
 */
export function AliyunAuthToolCard({
  tool,
  historical = false,
}: {
  tool: ToolUse;
  historical?: boolean;
}) {
  const { t } = useI18n();
  const notice = tool.notice;
  const showDialog = useAliyunDialog((s) => s.show);
  const [verifying, setVerifying] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; message: string } | null>(null);

  if (!notice || notice.kind !== "aliyun_authorization") return null;
  const bound = notice.bound;

  const onVerify = async () => {
    setVerifying(true);
    setResult(null);
    try {
      const r = await verifyAliyun();
      setResult(
        r.ok
          ? { ok: true, message: t("aliyun.card.verifyOk") }
          : {
              ok: false,
              message: r.verdict?.error_message || t("aliyun.card.verifyFail"),
            }
      );
    } catch (err) {
      setResult({
        ok: false,
        message: err instanceof Error ? err.message : t("aliyun.card.verifyError"),
      });
    } finally {
      setVerifying(false);
    }
  };

  const btn =
    "inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] px-3 py-1.5 text-xs font-semibold transition-colors";

  return (
    <div
      className={cn(
        "my-1 overflow-hidden rounded-[var(--radius-sm)] border",
        historical
          ? "border-[var(--border)] bg-[var(--surface-2)]/40"
          : "border-[var(--warning,var(--accent))]/30 bg-[var(--accent)]/5"
      )}
    >
      <div className="flex gap-2.5 px-3 py-2.5">
        <ShieldCheck
          className={cn(
            "mt-0.5 h-4 w-4 shrink-0",
            historical ? "text-[var(--text-faint)]" : "text-[var(--accent)]"
          )}
        />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <div
              className={cn(
                "text-xs font-semibold",
                historical ? "text-[var(--text-muted)]" : "text-[var(--text)]"
              )}
            >
              {bound ? t("aliyun.card.expiredTitle") : t("aliyun.card.needAuthTitle")}
            </div>
            {historical && (
              <span className="inline-flex items-center gap-1 rounded-full bg-[var(--surface-3)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-faint)]">
                <History className="h-3 w-3" />
                {t("aliyun.card.handled")}
              </span>
            )}
          </div>
          <div className="mt-0.5 text-xs leading-5 text-[var(--text-muted)]">
            {bound ? t("aliyun.card.expiredBody") : t("aliyun.card.needAuthBody")}
            {notice.error_code && (
              <span className="ml-1 font-mono text-[var(--text-faint)]">
                ({notice.error_code})
              </span>
            )}
          </div>

          {historical ? null : (
          <>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            {bound && (
              <button
                type="button"
                className={cn(
                  btn,
                  "border border-[var(--border)] text-[var(--text)] hover:bg-[var(--surface-2)]"
                )}
                onClick={() => void onVerify()}
                disabled={verifying}
              >
                {verifying ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : null}
                {t("aliyun.card.reverify")}
              </button>
            )}
            <button
              type="button"
              className={cn(btn, "bg-[var(--accent)] text-[var(--accent-fg)] hover:bg-[var(--accent-hover)]")}
              onClick={() => showDialog({ resumeAfter: true })}
            >
              {bound ? t("aliyun.card.reauthorize") : t("aliyun.card.authorize")}
            </button>
          </div>

          {result && (
            <div
              className={cn(
                "mt-2 flex items-start gap-1.5 rounded-[var(--radius-sm)] px-2.5 py-1.5 text-xs",
                result.ok
                  ? "bg-[var(--success)]/10 text-[var(--success)]"
                  : "bg-[var(--danger)]/10 text-[var(--danger)]"
              )}
            >
              {result.ok ? (
                <CircleCheck className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              ) : (
                <CircleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              )}
              <span className="min-w-0">{result.message}</span>
            </div>
          )}

          {result?.ok && (
            <button
              type="button"
              className={cn(
                btn,
                "mt-2 bg-[var(--accent)] text-[var(--accent-fg)] hover:bg-[var(--accent-hover)]"
              )}
              onClick={() => resumeAgent()}
            >
              {t("aliyun.card.continue")}
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          )}
          </>
          )}
        </div>
      </div>
    </div>
  );
}
