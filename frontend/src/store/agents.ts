import { create } from "zustand";
import { listAgents, type AgentSummary } from "../api/agents";
import { useChatStore } from "./chat";

interface AgentsState {
  agents: AgentSummary[];
  defaultAgent: string;
  loaded: boolean;
  loading: boolean;
  load: () => Promise<void>;
  /** Switch the active agent for the next turn and align the model dropdown:
   * an agent that pins a model sets it; one that leaves it blank hands model
   * choice back to the backend default (cleared here so ModelSelector re-resolves). */
  select: (id: string) => void;
  agentById: (id: string) => AgentSummary | undefined;
}

export const useAgentsStore = create<AgentsState>((set, get) => ({
  agents: [],
  defaultAgent: "main",
  loaded: false,
  loading: false,

  load: async () => {
    if (get().loading) return;
    set({ loading: true });
    try {
      const roster = await listAgents();
      set({ agents: roster.agents, defaultAgent: roster.default_agent, loaded: true });
      // Adopt the default agent on first load if the chat hasn't chosen one.
      const chat = useChatStore.getState();
      if (!chat.agentId) {
        const initial = roster.agents.find((a) => a.id === roster.default_agent) ?? roster.agents[0];
        if (initial) get().select(initial.id);
      }
    } catch {
      set({ loaded: true });
    } finally {
      set({ loading: false });
    }
  },

  select: (id) => {
    const agent = get().agents.find((a) => a.id === id);
    const chat = useChatStore.getState();
    chat.setAgent(id);
    chat.setModel(agent?.model || "");
  },

  agentById: (id) => get().agents.find((a) => a.id === id),
}));
