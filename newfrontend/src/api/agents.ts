import { apiFetch } from "../lib/apiFetch";

// A selectable agent, as exposed to any authenticated user by GET /v1/agents.
// (Managing agents stays admin-only via the config routes.)
export interface AgentSummary {
  id: string;
  name: string;
  description: string;
  model: string;
}

export interface AgentRoster {
  agents: AgentSummary[];
  default_agent: string;
}

export async function listAgents(): Promise<AgentRoster> {
  const res = await apiFetch("/v1/agents");
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body?.error?.message || body?.detail || `request failed: ${res.status}`);
  }
  return body as AgentRoster;
}
