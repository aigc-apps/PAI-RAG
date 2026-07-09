import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import { Check, ChevronRight, Database, Globe2, Lock, Terminal } from "lucide-react";
import { toast } from "sonner";
import type {
  AgentConfigDocument,
  CapabilityConfig,
  SetupMode,
} from "../api/agentConfig";
import { cn } from "../lib/cn";
import { useAgentConfigStore } from "../store/agentConfig";
import { BrandMark } from "./Sidebar";
import { ThemeToggle } from "./ThemeToggle";

const modes: Array<{
  id: SetupMode;
  title: string;
  body: string;
  points: string[];
}> = [
  {
    id: "local_first",
    title: "Local-first",
    body: "Start with local knowledge and minimal external dependencies.",
    points: ["Local knowledge ready", "Search skipped", "Sandbox disabled"],
  },
  {
    id: "cloud_enhanced",
    title: "Cloud-enhanced",
    body: "Use hosted search, embeddings, rerank, and vector databases.",
    points: ["Better retrieval", "Requires provider keys", "External services enabled"],
  },
  {
    id: "developer",
    title: "Developer mode",
    body: "Prepare the agent for script execution and automation workflows.",
    points: ["Sandbox-focused", "Approval controls", "Local or cloud runtime"],
  },
];

function statusLabel(cap?: CapabilityConfig) {
  if (!cap) return "Unknown";
  if (cap.status === "ready") return "Ready";
  if (cap.status === "missing_config") return "Needs setup";
  if (cap.status === "disabled") return "Disabled";
  return "Error";
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
  return (
    <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="mb-3 flex items-start gap-3">
        <div className="mt-0.5 rounded-[var(--radius-sm)] bg-[var(--surface-2)] p-2 text-[var(--text-muted)]">
          {icon}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-[var(--text)]">{cap?.name}</h3>
            <span className={cn("text-xs", statusClass(cap?.status))}>
              {statusLabel(cap)}
            </span>
          </div>
          <p className="mt-1 text-xs leading-5 text-[var(--text-muted)]">
            {cap?.description}
          </p>
        </div>
      </div>
      <div className="text-xs text-[var(--text-faint)]">
        Permission: <span className="text-[var(--text-muted)]">{cap?.permission}</span>
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

  const finish = async (skip = false) => {
    try {
      await completeSetup({
        completed: true,
        mode,
        skipped_steps: skip
          ? ["model", "search", "sandbox"]
          : [
              ...(caps.search?.status !== "ready" ? ["search"] : []),
              ...(caps.sandbox?.status !== "ready" ? ["sandbox"] : []),
            ],
      });
      onDone();
    } catch {
      toast.error("Could not save setup");
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
            First run setup
          </p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight">
            Configure your agent capabilities
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--text-muted)]">
            Start local-first, then enable search, cloud retrieval, sandbox execution,
            and skills as your deployment needs them.
          </p>
        </section>

        <section>
          <h2 className="mb-3 text-sm font-semibold">Deployment Mode</h2>
          <div className="grid gap-3 md:grid-cols-3">
            {modes.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setMode(item.id)}
                className={cn(
                  "min-h-[176px] rounded-[var(--radius)] border bg-[var(--surface)] p-4 text-left transition-colors",
                  mode === item.id
                    ? "border-[var(--accent)] bg-[var(--surface-2)]"
                    : "border-[var(--border)] hover:bg-[var(--surface-2)]"
                )}
              >
                <div className="mb-2 flex items-center justify-between">
                  <span className="font-semibold">{item.title}</span>
                  {mode === item.id && <Check className="h-4 w-4 text-[var(--accent)]" />}
                </div>
                <p className="mb-3 text-sm leading-5 text-[var(--text-muted)]">{item.body}</p>
                <div className="space-y-1 text-xs text-[var(--text-faint)]">
                  {item.points.map((point) => (
                    <div key={point}>{point}</div>
                  ))}
                </div>
              </button>
            ))}
          </div>
        </section>

        <section className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4">
          <div className="flex items-start gap-3">
            <div className="rounded-[var(--radius-sm)] bg-[var(--surface-2)] p-2">
              <Lock className="h-4 w-4 text-[var(--text-muted)]" />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-semibold">Model Provider</h2>
                <span className={cn("text-xs", llmReady ? "text-[var(--success)]" : "text-[var(--warning)]")}>
                  {llmReady ? "Ready" : "Needs setup"}
                </span>
              </div>
              <p className="mt-1 text-sm leading-6 text-[var(--text-muted)]">
                Current default model:{" "}
                <span className="font-mono text-xs text-[var(--text)]">
                  {String(providers["llm.default"]?.settings.default_model ?? "not configured")}
                </span>
              </p>
            </div>
          </div>
        </section>

        <section>
          <h2 className="mb-3 text-sm font-semibold">Core Capabilities</h2>
          <div className="grid gap-3 md:grid-cols-3">
            <CoreCard cap={caps.search} icon={<Globe2 className="h-4 w-4" />} />
            <CoreCard cap={caps.knowledge} icon={<Database className="h-4 w-4" />} />
            <CoreCard cap={caps.sandbox} icon={<Terminal className="h-4 w-4" />} />
          </div>
        </section>

        <section className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4">
          <h2 className="text-sm font-semibold">Recommended Skills</h2>
          <div className="mt-3 grid gap-2 md:grid-cols-2">
            {enabledSkills.map((skill) => (
              <div
                key={skill.id}
                className="flex items-center justify-between rounded-[var(--radius-sm)] bg-[var(--surface-2)] px-3 py-2"
              >
                <span className="text-sm">{skill.name}</span>
                <span className={cn("text-xs", statusClass(skill.status))}>
                  {statusLabel(skill)}
                </span>
              </div>
            ))}
          </div>
        </section>

        <section className="flex flex-col gap-3 border-t border-[var(--border)] pt-5 sm:flex-row sm:items-center">
          <div className="flex-1 text-xs leading-5 text-[var(--text-muted)]">
            Search defaults to approval, local knowledge can run automatically, and
            sandbox execution stays disabled until explicitly configured.
          </div>
          <button
            type="button"
            onClick={() => finish(true)}
            className="rounded-[var(--radius-sm)] px-3 py-2 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
          >
            Skip for now
          </button>
          <button
            type="button"
            disabled={loading}
            onClick={() => finish(false)}
            className="inline-flex items-center justify-center gap-2 rounded-[var(--radius-sm)] bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white disabled:opacity-60"
          >
            Finish setup
            <ChevronRight className="h-4 w-4" />
          </button>
        </section>
      </main>
    </div>
  );
}
