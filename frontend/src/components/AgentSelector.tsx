import { useEffect } from "react";
import { Bot } from "lucide-react";
import { useAgentsStore } from "../store/agents";
import { useChatStore } from "../store/chat";

/** Primary switcher on the chat surface: choose which agent answers. The chosen
 * agent pins its model (see agents store), while ModelSelector still allows a
 * per-turn override. Hidden until the roster loads / when only one agent exists
 * — a lone agent needs no picker but its model still applies. */
export function AgentSelector() {
  const agents = useAgentsStore((s) => s.agents);
  const loaded = useAgentsStore((s) => s.loaded);
  const load = useAgentsStore((s) => s.load);
  const select = useAgentsStore((s) => s.select);
  const agentId = useChatStore((s) => s.agentId);

  useEffect(() => { void load(); }, [load]);

  if (!loaded || agents.length <= 1) return null;

  const current = agentId || agents[0]?.id || "";
  return (
    <label className="flex items-center gap-1.5" title="选择智能体">
      <Bot className="h-4 w-4 text-[var(--text-muted)]" />
      <select
        aria-label="Agent"
        value={current}
        onChange={(e) => select(e.target.value)}
        className="text-xs font-medium rounded-[var(--radius-sm)] px-2 py-1 text-[var(--text)] bg-transparent hover:bg-[var(--surface-2)] border border-transparent focus:border-[var(--border-strong)] outline-none cursor-pointer transition-colors"
      >
        {agents.map((a) => (
          <option key={a.id} value={a.id}>{a.name}</option>
        ))}
      </select>
    </label>
  );
}
