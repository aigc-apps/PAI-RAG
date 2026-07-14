import { create } from "zustand";
import {
  getAgentConfig,
  getAgentConfigYaml,
  enableSkillForAgent,
  getSetup,
  installSkill,
  saveAgentConfig,
  saveAgentConfigYaml,
  saveSetup,
  testSearchProvider,
  uninstallSkill,
  uploadSkillZip,
  type AgentConfigDocument,
  type SkillInstallSource,
  type SetupConfig,
} from "../api/agentConfig";

interface AgentConfigState {
  doc?: AgentConfigDocument;
  loading: boolean;
  error?: string;
  loadSetup: () => Promise<AgentConfigDocument | undefined>;
  refresh: () => Promise<AgentConfigDocument | undefined>;
  completeSetup: (setup: SetupConfig) => Promise<AgentConfigDocument>;
  save: (doc: AgentConfigDocument) => Promise<AgentConfigDocument>;
  loadYaml: () => Promise<string>;
  saveYaml: (yaml: string) => Promise<AgentConfigDocument>;
  testSearch: (query: string, numResults?: number) => Promise<{ ok: boolean; output: string }>;
  uploadSkillZip: (file: File) => Promise<{ upload_id: string; filename?: string; size: number }>;
  installSkill: (payload: {
    source: SkillInstallSource;
    enable_for_agent?: string;
    enable_after_build?: boolean;
    overwrite?: boolean;
  }) => Promise<AgentConfigDocument>;
  enableSkillForAgent: (payload: {
    skill_id: string;
    agent_id?: string;
    enabled?: boolean;
  }) => Promise<AgentConfigDocument>;
  uninstallSkill: (payload: {
    skill_id: string;
    confirm: boolean;
  }) => Promise<AgentConfigDocument>;
}

export const useAgentConfigStore = create<AgentConfigState>((set) => ({
  doc: undefined,
  loading: false,
  error: undefined,

  loadSetup: async () => {
    set({ loading: true, error: undefined });
    try {
      const doc = await getSetup();
      set({ doc, loading: false });
      return doc;
    } catch (err) {
      const error = err instanceof Error ? err.message : "setup load failed";
      set({ error, loading: false });
      return undefined;
    }
  },

  refresh: async () => {
    set({ loading: true, error: undefined });
    try {
      const doc = await getAgentConfig();
      set({ doc, loading: false });
      return doc;
    } catch (err) {
      const error = err instanceof Error ? err.message : "config load failed";
      set({ error, loading: false });
      return undefined;
    }
  },

  completeSetup: async (setup) => {
    set({ loading: true, error: undefined });
    try {
      const doc = await saveSetup(setup);
      set({ doc, loading: false });
      return doc;
    } catch (err) {
      const error = err instanceof Error ? err.message : "setup save failed";
      set({ error, loading: false });
      throw err;
    }
  },

  save: async (doc) => {
    set({ loading: true, error: undefined });
    try {
      const saved = await saveAgentConfig(doc);
      set({ doc: saved, loading: false });
      return saved;
    } catch (err) {
      const error = err instanceof Error ? err.message : "config save failed";
      set({ error, loading: false });
      throw err;
    }
  },

  loadYaml: async () => getAgentConfigYaml(),

  saveYaml: async (yaml) => {
    set({ loading: true, error: undefined });
    try {
      const saved = await saveAgentConfigYaml(yaml);
      set({ doc: saved, loading: false });
      return saved;
    } catch (err) {
      const error = err instanceof Error ? err.message : "YAML save failed";
      set({ error, loading: false });
      throw err;
    }
  },

  testSearch: async (query, numResults = 3) =>
    testSearchProvider(query, numResults),

  uploadSkillZip: async (file) => uploadSkillZip(file),

  installSkill: async (payload) => {
    set({ loading: true, error: undefined });
    try {
      const result = await installSkill(payload);
      set({ doc: result.config, loading: false });
      return result.config;
    } catch (err) {
      const error = err instanceof Error ? err.message : "skill install failed";
      set({ error, loading: false });
      throw err;
    }
  },

  enableSkillForAgent: async (payload) => {
    set({ loading: true, error: undefined });
    try {
      const result = await enableSkillForAgent(payload);
      set({ doc: result.config, loading: false });
      return result.config;
    } catch (err) {
      const error = err instanceof Error ? err.message : "skill enable failed";
      set({ error, loading: false });
      throw err;
    }
  },

  uninstallSkill: async (payload) => {
    set({ loading: true, error: undefined });
    try {
      const result = await uninstallSkill(payload);
      set({ doc: result.config, loading: false });
      return result.config;
    } catch (err) {
      const error = err instanceof Error ? err.message : "skill uninstall failed";
      set({ error, loading: false });
      throw err;
    }
  },
}));
