import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import { Check, ChevronRight, CircleCheck, CircleAlert, Database, Globe2, Terminal } from "lucide-react";
import { toast } from "sonner";
import type {
  AgentConfigDocument,
  CapabilityConfig,
  SetupMode,
} from "../api/agentConfig";
import { cn } from "../lib/cn";
import { CARD, BTN_PRIMARY } from "../lib/ui";
import { useAgentConfigStore } from "../store/agentConfig";
import { BrandMark } from "./Sidebar";
import { ConnectionsPanel } from "./ConnectionsPanel";
import { ThemeToggle } from "./ThemeToggle";
import { useI18n, type MessageKey, type TFunction } from "../i18n";

const modes: Array<{
  id: SetupMode;
  titleKey: MessageKey;
  bodyKey: MessageKey;
  pointKeys: MessageKey[];
}> = [
  {
    id: "local_first",
    titleKey: "setup.mode.localFirst.title",
    bodyKey: "setup.mode.localFirst.body",
    pointKeys: ["setup.mode.localFirst.p1", "setup.mode.localFirst.p2", "setup.mode.localFirst.p3"],
  },
  {
    id: "cloud_enhanced",
    titleKey: "setup.mode.cloudEnhanced.title",
    bodyKey: "setup.mode.cloudEnhanced.body",
    pointKeys: ["setup.mode.cloudEnhanced.p1", "setup.mode.cloudEnhanced.p2", "setup.mode.cloudEnhanced.p3"],
  },
  {
    id: "developer",
    titleKey: "setup.mode.developer.title",
    bodyKey: "setup.mode.developer.body",
    pointKeys: ["setup.mode.developer.p1", "setup.mode.developer.p2", "setup.mode.developer.p3"],
  },
];

function statusLabel(t: TFunction, cap?: CapabilityConfig): string {
  if (!cap) return t("setup.status.unknown");
  if (cap.status === "ready") return t("setup.status.ready");
  if (cap.status === "missing_config") return t("setup.status.needsSetup");
  if (cap.status === "disabled") return t("setup.status.disabled");
  return t("setup.status.error");
}

function statusClass(status?: string) {
  if (status === "ready") return "text-[var(--success)]";
  if (status === "missing_config") return "text-[var(--warning)]";
  if (status === "error") return "text-[var(--danger)]";
  return "text-[var(--text-faint)]";
}

function CoreCard({
  cap,
  icon,
}: {
  cap?: CapabilityConfig;
  icon: ReactNode;
}) {
  const { t } = useI18n();
  return (
    <div className={CARD}>
      <div className="mb-3 flex items-start gap-3">
        <div className="mt-0.5 rounded-[var(--radius-sm)] bg-[var(--surface-2)] p-2 text-[var(--text-muted)]">
          {icon}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-[var(--text)]">{cap?.name}</h3>
            <span className={cn("text-xs", statusClass(cap?.status))}>
              {statusLabel(t, cap)}
            </span>
          </div>
          <p className="mt-1 text-xs leading-5 text-[var(--text-muted)]">
            {cap?.description}
          </p>
        </div>
      </div>
      <div className="text-xs text-[var(--text-faint)]">
        {t("setup.permission")} <span className="text-[var(--text-muted)]">{cap?.permission}</span>
      </div>
    </div>
  );
}

export function SetupWizard({
  doc,
  onDone,
}: {
  doc: AgentConfigDocument;
  onDone: () => void;
}) {
  const { t } = useI18n();
  const [mode, setMode] = useState<SetupMode>(doc.setup.mode ?? "local_first");
  const completeSetup = useAgentConfigStore((s) => s.completeSetup);
  const loading = useAgentConfigStore((s) => s.loading);
  const caps = useMemo(
    () => Object.fromEntries(doc.capabilities.map((cap) => [cap.id, cap])),
    [doc.capabilities]
  );
  const providers = useMemo(
    () => Object.fromEntries(doc.providers.map((provider) => [provider.id, provider])),
    [doc.providers]
  );
  const llmReady = providers["llm.default"]?.status === "healthy";
  const enabledSkills = doc.capabilities.filter(
    (cap) => cap.kind === "skill" && cap.enabled
  );

  // Finish is gated on a healthy model (the one required step); the optional
  // capabilities record themselves as skipped so Control Room can prompt later.
  const finish = async () => {
    try {
      await completeSetup({
        completed: true,
        mode,
        skipped_steps: [
          ...(caps.search?.status !== "ready" ? ["search"] : []),
          ...(caps.sandbox?.status !== "ready" ? ["sandbox"] : []),
        ],
      });
      onDone();
    } catch {
      toast.error(t("setup.saveFailed"));
    }
  };

  return (
    <div className="min-h-full bg-[var(--bg)] text-[var(--text)]">
      <div className="flex h-12 items-center border-b border-[var(--border)] px-4">
        <BrandMark />
        <div className="flex-1" />
        <ThemeToggle />
      </div>
      <main className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-5 py-8">
        <section>
          <p className="text-xs font-medium uppercase tracking-[0.18em] text-[var(--text-faint)]">
            {t("setup.firstRun")}
          </p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight">
            {t("setup.configureTitle")}
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--text-muted)]">
            {t("setup.introBody")}
          </p>
        </section>

        <section>
          <h2 className="mb-3 text-sm font-semibold">{t("setup.deploymentMode")}</h2>
          <div className="grid gap-3 md:grid-cols-3">
            {modes.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setMode(item.id)}
                className={cn(
                  "min-h-[176px] rounded-[var(--radius-lg)] border bg-[var(--bg-elevated)] p-4 text-left transition-colors",
                  mode === item.id
                    ? "border-[var(--accent)] bg-[var(--surface-2)]"
                    : "border-[var(--border)] hover:bg-[var(--surface-2)]"
                )}
              >
                <div className="mb-2 flex items-center justify-between">
                  <span className="font-semibold">{t(item.titleKey)}</span>
                  {mode === item.id && <Check className="h-4 w-4 text-[var(--accent)]" />}
                </div>
                <p className="mb-3 text-sm leading-5 text-[var(--text-muted)]">{t(item.bodyKey)}</p>
                <div className="space-y-1 text-xs text-[var(--text-faint)]">
                  {item.pointKeys.map((pointKey) => (
                    <div key={pointKey}>{t(pointKey)}</div>
                  ))}
                </div>
              </button>
            ))}
          </div>
        </section>

        <section className={CARD}>
          <div className="mb-4 flex items-center gap-2">
            <h2 className="text-sm font-semibold">{t("setup.connectModel")}</h2>
            <span
              className={cn(
                "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium",
                llmReady
                  ? "bg-[var(--success)]/12 text-[var(--success)]"
                  : "bg-[var(--warning)]/12 text-[var(--warning)]"
              )}
            >
              {llmReady ? (
                <CircleCheck className="h-3.5 w-3.5" />
              ) : (
                <CircleAlert className="h-3.5 w-3.5" />
              )}
              {llmReady ? t("setup.status.ready") : t("setup.required")}
            </span>
          </div>
          <p className="mb-4 max-w-2xl text-sm leading-6 text-[var(--text-muted)]">
            {t("setup.connectModelBody")}
          </p>
          <ConnectionsPanel doc={doc} />
        </section>

        <section>
          <h2 className="mb-3 text-sm font-semibold">{t("setup.coreCapabilities")}</h2>
          <div className="grid gap-3 md:grid-cols-3">
            <CoreCard cap={caps.search} icon={<Globe2 className="h-4 w-4" />} />
            <CoreCard cap={caps.knowledge} icon={<Database className="h-4 w-4" />} />
            <CoreCard cap={caps.sandbox} icon={<Terminal className="h-4 w-4" />} />
          </div>
        </section>

        <section className={CARD}>
          <h2 className="text-sm font-semibold">{t("setup.recommendedSkills")}</h2>
          <div className="mt-3 grid gap-2 md:grid-cols-2">
            {enabledSkills.map((skill) => (
              <div
                key={skill.id}
                className="flex items-center justify-between rounded-[var(--radius-sm)] bg-[var(--surface-2)] px-3 py-2"
              >
                <span className="text-sm">{skill.name}</span>
                <span className={cn("text-xs", statusClass(skill.status))}>
                  {statusLabel(t, skill)}
                </span>
              </div>
            ))}
          </div>
        </section>

        <section className="flex flex-col gap-3 border-t border-[var(--border)] pt-5 sm:flex-row sm:items-center">
          <div className="flex-1 text-xs leading-5 text-[var(--text-muted)]">
            {llmReady ? t("setup.finishHintReady") : t("setup.finishHintNeedModel")}
          </div>
          <button
            type="button"
            disabled={loading || !llmReady}
            title={llmReady ? undefined : t("setup.finishTitleGate")}
            onClick={() => finish()}
            className={cn(BTN_PRIMARY, "px-4")}
          >
            {t("setup.finishSetup")}
            <ChevronRight className="h-4 w-4" />
          </button>
        </section>
      </main>
    </div>
  );
}
