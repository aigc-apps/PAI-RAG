import { useEffect, useState } from "react";
import { Bot } from "lucide-react";
import { useAgentsStore } from "../store/agents";
import { useChatStore } from "../store/chat";
import { listModels } from "../api/models";
import type { AgentSummary } from "../api/agents";
import { useI18n } from "../i18n";

/** Short, human model name: drop the "provider/" prefix ("openai/gpt-4o" → "gpt-4o"). */
function shortModel(model: string): string {
  return model.includes("/") ? model.slice(model.lastIndexOf("/") + 1) : model;
}

/** Chat-surface agent control. Shows the active agent and the model it resolves to
 * — e.g. "Main (gpt-4o)". With several agents it's a switcher; with a single agent
 * it's just a label. There is no separate model selector: an agent's own model (or
 * the deployment default when it inherits) is what runs, so we surface it here
 * rather than letting the user override the model per turn. */
export function AgentSelector() {
  const { t } = useI18n();
  const agents = useAgentsStore((s) => s.agents);
  const loaded = useAgentsStore((s) => s.loaded);
  const load = useAgentsStore((s) => s.load);
  const select = useAgentsStore((s) => s.select);
  const agentId = useChatStore((s) => s.agentId);
  const [defaultModel, setDefaultModel] = useState("");

  useEffect(() => {
    void load();
  }, [load]);

  // The resolved deployment default, shown in parens when an agent inherits it.
  useEffect(() => {
    let cancelled = false;
    listModels()
      .then(({ default: def }) => {
        if (!cancelled) setDefaultModel(def);
      })
      .catch(() => {
        /* leave blank; the agent's own model still shows */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!loaded || agents.length === 0) return null;

  const current = agents.find((a) => a.id === agentId) ?? agents[0];
  const label = (a: AgentSummary) =>
    `${a.name} (${shortModel(a.model || defaultModel) || t("agent.defaultModel")})`;

  // A lone agent needs no picker — just show what's running.
  if (agents.length === 1) {
    return (
      <span
        className="flex items-center gap-1.5 text-xs font-medium text-[var(--text-muted)]"
        title={t("agent.current")}
      >
        <Bot className="h-4 w-4" />
        {label(current)}
      </span>
    );
  }

  return (
    <label className="flex items-center gap-1.5" title={t("agent.select")}>
      <Bot className="h-4 w-4 text-[var(--text-muted)]" />
      <select
        aria-label={t("agent.aria")}
        value={current.id}
        onChange={(e) => select(e.target.value)}
        className="text-xs font-medium rounded-[var(--radius-sm)] px-2 py-1 text-[var(--text)] bg-transparent hover:bg-[var(--surface-2)] border border-transparent focus:border-[var(--border-strong)] outline-none cursor-pointer transition-colors"
      >
        {agents.map((a) => (
          <option key={a.id} value={a.id}>
            {label(a)}
          </option>
        ))}
      </select>
    </label>
  );
}
