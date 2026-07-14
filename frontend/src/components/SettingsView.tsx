import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  CircleAlert,
  Database,
  GitBranch,
  Globe2,
  Loader2,
  Pencil,
  Plus,
  Settings2,
  ShieldCheck,
  Sparkles,
  Terminal,
  Trash2,
  Upload,
  UserRound,
  Wrench,
  X,
} from "lucide-react";
import { toast } from "sonner";
import type {
  AgentProfile,
  CapabilityConfig,
  AgentConfigDocument,
} from "../api/agentConfig";
import { generateCodeManifest, newAgentProfile } from "../api/agentConfig";
import { listKnowledgeBases } from "../api/knowledge";
import type { KnowledgeBase } from "../api/knowledge";
import { cn } from "../lib/cn";
import { CARD, INPUT, BTN_PRIMARY, BTN_GHOST, BTN_DANGER } from "../lib/ui";
import { useAgentConfigStore } from "../store/agentConfig";
import { useAliyunDialog } from "../store/aliyunDialog";
import { ConnectionsPanel } from "./ConnectionsPanel";
import { KnowledgeView, type KnowledgeTab } from "./KnowledgeView";
import { PageHeader } from "./PageHeader";
import { useI18n } from "../i18n";

export type SettingsSection =
  | "agents"
  | "org-persona"
  | "tools"
  | "connections"
  | "skills"
  | "knowledge"
  | "yaml";
type TabItem = { id: SettingsSection; label: string; subtle?: boolean };

function isControlPlaneCapability(cap: CapabilityConfig) {
  return cap.settings.control_plane === true;
}

function statusClass(status: string) {
  if (status === "ready" || status === "healthy") return "text-[var(--success)]";
  if (status === "missing_config") return "text-[var(--warning)]";
  if (status === "error") return "text-[var(--danger)]";
  return "text-[var(--text-faint)]";
}

function statusLabel(cap: CapabilityConfig) {
  if (cap.status === "ready") return "Ready";
  if (cap.status === "missing_config") return "Needs setup";
  if (cap.status === "disabled") return "Disabled";
  return "Error";
}

function skillMeta(skill: CapabilityConfig) {
  const source = skill.settings.source ? String(skill.settings.source) : "";
  if (!source) return "";
  const version = skill.settings.version ? ` · v${String(skill.settings.version)}` : "";
  const path = skill.settings.path ? ` · ${String(skill.settings.path)}` : "";
  return `${source}${version}${path}`;
}

function iconFor(id: string) {
  if (id === "search") return <Globe2 className="h-4 w-4" />;
  if (id === "knowledge") return <Database className="h-4 w-4" />;
  if (id === "sandbox") return <Terminal className="h-4 w-4" />;
  if (id === "aliyun_pai") return <ShieldCheck className="h-4 w-4" />;
  return <Wrench className="h-4 w-4" />;
}

function displayToolName(id: string) {
  if (id === "search") return "web_search";
  if (id === "knowledge") return "knowledge_search";
  if (id === "sandbox") return "code_sandbox";
  return id;
}

const KNOWLEDGE_TOOL_BUNDLE = [
  "knowledge_search",
  "knowledge_read",
  "knowledge_find",
  "knowledge_list",
];

const LEGACY_KNOWLEDGE_TOOL_NAMES = [
  "view_file",
  "grep_file",
  "list_knowledge_bases",
];

const TOOL_BUNDLES: Record<string, string[]> = {
  knowledge_search: KNOWLEDGE_TOOL_BUNDLE,
  code_sandbox: ["code_interpreter", "shell", "publish_artifact"],
  // The `subagent` capability id maps to the real registered tool name, so toggling
  // the "Subagents" card writes spawn_subagent into the agent's tools.include
  // (mirrors the sandbox → code_sandbox bundle). Parallel fan-out is not a separate
  // tool — the model emits several spawn_subagent calls and the loop runs them
  // concurrently — so there is just one name to bundle here.
  subagent: ["spawn_subagent"],
};

function toolBundle(toolId: string) {
  return TOOL_BUNDLES[toolId] ?? [toolId];
}

function toolAliases(toolId: string) {
  const legacyAliases = toolId === "knowledge_search" ? LEGACY_KNOWLEDGE_TOOL_NAMES : [];
  return Array.from(new Set([toolId, ...toolBundle(toolId), ...legacyAliases]));
}

function persistedToolNames(toolId: string) {
  return Array.from(new Set([toolId, ...toolBundle(toolId)]));
}

function isToolEnabled(agent: AgentProfile, toolId: string) {
  const included = new Set(agent.tools.include);
  return toolAliases(toolId).some((name) => included.has(name));
}

function enabledToolIds(agent: AgentProfile, tools: ReturnType<typeof systemTools>) {
  return new Set(tools.filter((tool) => isToolEnabled(agent, tool.id)).map((tool) => tool.id));
}

function systemTools(doc: AgentConfigDocument) {
  const core = doc.capabilities
    // aliyun_pai is a deployment capability (governs sandbox credential
    // injection), not a callable tool an agent selects — skip it here.
    .filter(
      (cap) =>
        cap.kind === "core_tool" &&
        cap.id !== "aliyun_pai" &&
        !isControlPlaneCapability(cap)
    )
    .map((cap) => ({
      id: displayToolName(cap.id),
      sourceId: cap.id,
      name: displayToolName(cap.id),
      available: cap.status === "ready",
      status: cap.status,
      description: cap.description,
    }));
  return [
    { id: "current_datetime", sourceId: "current_datetime", name: "current_datetime", available: true, status: "ready", description: "Current date and time." },
    { id: "web_fetch", sourceId: "web_fetch", name: "web_fetch", available: true, status: "ready", description: "Fetch and extract readable text from a URL." },
    ...core,
  ];
}

function skillSummary(doc: AgentConfigDocument, agent: AgentProfile) {
  const enabled = new Set(agent.skills.enabled);
  return doc.capabilities.filter((cap) => cap.kind === "skill" && enabled.has(cap.id));
}

function applyAgentPatch(
  doc: AgentConfigDocument,
  agentId: string,
  patch: Partial<AgentProfile>
): AgentConfigDocument {
  return {
    ...doc,
    agents: doc.agents.map((agent) =>
      agent.id === agentId ? { ...agent, ...patch } : agent
    ),
  };
}

export function SettingsView({
  doc,
  onBack,
  initialSection = "agents",
  onSectionChange,
}: {
  doc: AgentConfigDocument;
  onBack: () => void;
  initialSection?: SettingsSection;
  onSectionChange?: (section: SettingsSection) => void;
}) {
  const save = useAgentConfigStore((s) => s.save);
  const loadYaml = useAgentConfigStore((s) => s.loadYaml);
  const saveYaml = useAgentConfigStore((s) => s.saveYaml);
  const testSearch = useAgentConfigStore((s) => s.testSearch);
  const uploadSkillZip = useAgentConfigStore((s) => s.uploadSkillZip);
  const installSkill = useAgentConfigStore((s) => s.installSkill);
  const loading = useAgentConfigStore((s) => s.loading);
  const [tab, setTab] = useState<SettingsSection>(initialSection);
  const [agentId, setAgentId] = useState(doc.default_agent || doc.agents[0]?.id || "main");
  const [searchOpen, setSearchOpen] = useState(false);
  const [sandboxOpen, setSandboxOpen] = useState(false);
  const [vectordbOpen, setVectordbOpen] = useState(false);
  const showAliyunDialog = useAliyunDialog((s) => s.show);
  const [skillInstallOpen, setSkillInstallOpen] = useState(false);
  const [yamlText, setYamlText] = useState("");
  const [testingSearch, setTestingSearch] = useState(false);
  const [searchOutput, setSearchOutput] = useState("");

  useEffect(() => {
    setTab(initialSection);
  }, [initialSection]);

  const selectSection = (section: SettingsSection) => {
    setTab(section);
    onSectionChange?.(section);
  };

  const agents = doc.agents.length ? doc.agents : [];
  const agent = agents.find((item) => item.id === agentId) ?? agents[0];
  const tools = useMemo(() => systemTools(doc), [doc]);
  const coreTools = doc.capabilities.filter(
    (cap) => cap.kind === "core_tool" && !isControlPlaneCapability(cap)
  );
  const skills = doc.capabilities.filter((cap) => cap.kind === "skill");

  const saveDoc = async (next: AgentConfigDocument, message = "Could not save") => {
    try {
      await save(next);
    } catch {
      toast.error(message);
    }
  };

  const toggleAgentTool = (toolId: string) => {
    if (!agent) return;
    const included = new Set(agent.tools.include);
    const excluded = new Set(agent.tools.exclude);
    const aliases = toolAliases(toolId);
    const persistedNames = persistedToolNames(toolId);
    if (isToolEnabled(agent, toolId)) {
      aliases.forEach((name) => included.delete(name));
      aliases.forEach((name) => excluded.delete(name));
      persistedNames.forEach((name) => excluded.add(name));
    } else {
      toolBundle(toolId).forEach((name) => included.add(name));
      aliases.forEach((name) => excluded.delete(name));
    }
    void saveDoc(
      applyAgentPatch(doc, agent.id, {
        tools: { include: [...included], exclude: [...excluded] },
      })
    );
  };

  const toggleAgentSkill = (skillId: string) => {
    if (!agent) return;
    const enabled = new Set(agent.skills.enabled);
    if (enabled.has(skillId)) enabled.delete(skillId);
    else enabled.add(skillId);
    void saveDoc(
      applyAgentPatch(doc, agent.id, {
        skills: { enabled: [...enabled] },
      })
    );
  };

  const patchCapability = async (id: string, patch: Partial<CapabilityConfig>) => {
    await saveDoc({
      ...doc,
      capabilities: doc.capabilities.map((cap) =>
        cap.id === id ? { ...cap, ...patch } : cap
      ),
    });
  };

  const openYaml = async () => {
    try {
      setYamlText(await loadYaml());
      selectSection("yaml");
    } catch {
      toast.error("Could not load YAML");
    }
  };

  const runSearchTest = async () => {
    setTestingSearch(true);
    setSearchOutput("");
    try {
      const result = await testSearch("OpenAI latest news", 3);
      setSearchOutput(result.output);
      if (result.ok) toast.success("Search provider works");
      else toast.error("Search returned an error");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Search test failed");
    } finally {
      setTestingSearch(false);
    }
  };

  const { t } = useI18n();
  const tabs: TabItem[] = [
    { id: "agents", label: t("settings.navAgents") },
    { id: "connections", label: t("settings.navConnections") },
    { id: "tools", label: t("settings.navCapabilities") },
    { id: "knowledge", label: t("settings.navKnowledge") },
    { id: "skills", label: t("settings.navSkills") },
    { id: "org-persona", label: t("settings.navDefaultPersona"), subtle: true },
    { id: "yaml", label: t("settings.navYaml"), subtle: true },
  ];

  return (
    <div className="workspace-page flex h-full flex-col text-[var(--text)]">
      <PageHeader
        icon={Settings2}
        title={t("settings.title")}
        onBack={onBack}
        backLabel={t("users.backToChat")}
      />

      <main
        className="mx-auto grid w-full max-w-[1160px] flex-1 grid-cols-[196px_1fr] gap-6 overflow-y-auto px-7 py-6 max-lg:grid-cols-1 max-md:px-4 max-md:py-5"
      >
        <aside className="space-y-1 max-lg:flex max-lg:gap-1 max-lg:overflow-x-auto max-lg:pb-1">
          {tabs.map((item, index) => {
            return (
              <button
                key={item.id}
                type="button"
                onClick={async () => {
                  if (item.id === "yaml" && !yamlText) await openYaml();
                  else selectSection(item.id);
                }}
                className={cn(
                  "flex min-h-9 w-full items-center rounded-[var(--radius)] px-3 py-1.5 text-left text-sm transition-colors max-lg:w-auto max-lg:shrink-0",
                  item.subtle && index > 0 && "mt-2 max-lg:mt-0",
                  tab === item.id
                    ? "bg-[var(--bg-elevated)] text-[var(--text)] font-semibold shadow-[var(--shadow-sm)]"
                    : item.subtle
                      ? "text-[var(--text-faint)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text)]"
                      : "text-[var(--text-muted)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text)]"
                )}
              >
                {item.label}
              </button>
            );
          })}
        </aside>

        <section className="min-w-0">
          {tab === "agents" && agent && (
            <AgentsPanel
              doc={doc}
              agent={agent}
              agents={agents}
              setSelectedAgentId={setAgentId}
              tools={tools}
              skills={skills}
              loading={loading}
              onSave={saveDoc}
              onToggleTool={toggleAgentTool}
              onToggleSkill={toggleAgentSkill}
            />
          )}

          {tab === "org-persona" && (
            <OrgPersonaPanel doc={doc} onSave={saveDoc} />
          )}

          {tab === "tools" && (
            <ToolsPanel
              tools={coreTools}
              loading={loading}
              onConfigureSearch={() => setSearchOpen(true)}
              onConfigureSandbox={() => setSandboxOpen(true)}
              onConfigureAliyun={() => showAliyunDialog()}
              onConfigureVectorDB={() => setVectordbOpen(true)}
              onPatchCapability={patchCapability}
            />
          )}

          {tab === "connections" && <ConnectionsPanel doc={doc} />}

          {tab === "knowledge" && <KnowledgeSettingsPanel />}

          {tab === "skills" && (
            <SkillsPanel
              skills={skills}
              loading={loading}
              onInstall={() => setSkillInstallOpen(true)}
              onPatchCapability={patchCapability}
            />
          )}

          {tab === "yaml" && (
            <YamlPanel
              yamlText={yamlText}
              setYamlText={setYamlText}
              loading={loading}
              onLoad={openYaml}
              onSave={async () => {
                try {
                  await saveYaml(yamlText);
                  toast.success("Saved YAML");
                } catch (err) {
                  const msg = err instanceof Error ? err.message : "Invalid YAML or save failed";
                  toast.error(msg);
                  // eslint-disable-next-line no-console
                  console.error("save yaml failed:", msg);
                }
              }}
            />
          )}
        </section>
      </main>

      {searchOpen && (
        <SearchConfigDialog
          doc={doc}
          loading={loading}
          onClose={() => setSearchOpen(false)}
          onSave={async (next) => {
            try {
              await save(next);
              setSearchOpen(false);
            } catch {
              toast.error("Could not save search configuration");
            }
          }}
          onTest={runSearchTest}
          testing={testingSearch}
          output={searchOutput}
        />
      )}

      {sandboxOpen && (
        <SandboxConfigDialog
          doc={doc}
          loading={loading}
          onClose={() => setSandboxOpen(false)}
          onSave={async (next) => {
            try {
              await save(next);
              setSandboxOpen(false);
            } catch {
              toast.error("Could not save sandbox configuration");
            }
          }}
        />
      )}

      {vectordbOpen && (
        <VectorDBConfigDialog
          doc={doc}
          loading={loading}
          onClose={() => setVectordbOpen(false)}
          onSave={async (next) => {
            try {
              await save(next);
              setVectordbOpen(false);
            } catch {
              toast.error("Could not save vector database configuration");
            }
          }}
        />
      )}

      {skillInstallOpen && (
        <SkillInstallDialog
          agents={agents}
          loading={loading}
          onClose={() => setSkillInstallOpen(false)}
          onUpload={uploadSkillZip}
          onInstall={async (payload) => {
            try {
              await installSkill(payload);
              toast.success("Skill installed");
              setSkillInstallOpen(false);
            } catch (err) {
              toast.error(err instanceof Error ? err.message : "Skill install failed");
            }
          }}
        />
      )}
    </div>
  );
}

/** Knowledge-base management, inlined into the Settings panel so it behaves like
 * every other tab (no full-page navigation). KB list ↔ detail and the detail's
 * sub-tabs are driven by local state — mirroring how the Agents tab keeps its
 * selected agent and open dialogs local rather than in the URL. */
function KnowledgeSettingsPanel() {
  const [kbId, setKbId] = useState<string | undefined>(undefined);
  const [kbTab, setKbTab] = useState<KnowledgeTab>("overview");
  return (
    <KnowledgeView
      embedded
      kbId={kbId}
      tab={kbTab}
      onOpenKb={(id) => {
        setKbId(id);
        setKbTab("overview");
      }}
      onBackToList={() => setKbId(undefined)}
      onTabChange={setKbTab}
      onInvalidKb={() => setKbId(undefined)}
      onBack={() => {}}
    />
  );
}

function Header({ title, body }: { title: string; body: string }) {
  return (
    <div className="mb-5">
      <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--text-muted)]">
        {body}
      </p>
    </div>
  );
}

/** Shared modal shell for the per-agent editors, matching the app's existing
 * dialogs (backdrop + header + X). `wide` widens it for the persona editor. */
function EditorDialog({
  title,
  onClose,
  wide,
  footer,
  children,
}: {
  title: string;
  onClose: () => void;
  wide?: boolean;
  footer?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-black/30 p-4"
      onMouseDown={(e) => {
        // Backdrop click (not a drag ending outside) dismisses.
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className={cn(
          "flex max-h-[85vh] w-full flex-col rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-elevated)] shadow-xl",
          wide ? "max-w-3xl" : "max-w-xl"
        )}
      >
        <div className="flex h-11 shrink-0 items-center border-b border-[var(--border)] px-4">
          <div className="text-sm font-semibold">{title}</div>
          <button
            type="button"
            aria-label="Close"
            onClick={onClose}
            className="ml-auto rounded-[var(--radius-sm)] p-1 text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-4">{children}</div>
        {footer && (
          <div className="flex shrink-0 items-center justify-end gap-2 border-t border-[var(--border)] px-4 py-3">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}

/** Persona editor — the agent's full system prompt (freeform Markdown). This IS
 * the agent's persona/base prompt; tools and skills are appended automatically.
 * Local state; "保存" commits via the whole-doc PUT and closes, X/backdrop discards. */
function PersonaDialog({
  doc,
  agent,
  onClose,
  onSave,
}: {
  doc: AgentConfigDocument;
  agent: AgentProfile;
  onClose: () => void;
  onSave: (doc: AgentConfigDocument, message?: string) => Promise<void>;
}) {
  const { t } = useI18n();
  const [text, setText] = useState<string>(agent.instructions ?? "");
  const dirty = text !== (agent.instructions ?? "");

  const save = () => {
    if (dirty) void onSave(applyAgentPatch(doc, agent.id, { instructions: text }));
    onClose();
  };

  return (
    <EditorDialog
      title={`Persona — ${agent.name}`}
      onClose={onClose}
      wide
      footer={
        <>
          <button type="button" onClick={onClose} className={BTN_GHOST}>
            {t("common.cancel")}
          </button>
          <button type="button" onClick={save} disabled={!dirty} className={BTN_PRIMARY}>
            {t("common.save")}
          </button>
        </>
      }
    >
      <p className="mb-3 text-xs text-[var(--text-muted)]">
        {t("settings.agentPersonaBody")}
      </p>
      <textarea
        autoFocus
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={20}
        placeholder={t("settings.defaultPersonaPlaceholder")}
        className="w-full resize-y rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 font-mono text-xs leading-6"
      />
    </EditorDialog>
  );
}

/** The tool-toggle grid, reused inside ToolsDialog. Toggles persist immediately. */
function ToolsGrid({
  tools,
  enabled,
  loading,
  onToggle,
}: {
  tools: ReturnType<typeof systemTools>;
  enabled: Set<string>;
  loading: boolean;
  onToggle: (toolId: string) => void;
}) {
  return (
    <div className="grid gap-2 md:grid-cols-2">
      {tools.map((tool) => (
        <button
          key={tool.id}
          type="button"
          disabled={loading || !tool.available}
          onClick={() => onToggle(tool.id)}
          className={cn(
            "rounded-[var(--radius-sm)] border px-3 py-2 text-left text-sm",
            enabled.has(tool.id)
              ? "border-[var(--accent)] bg-[var(--surface-2)] text-[var(--text)]"
              : "border-[var(--border)] text-[var(--text-muted)] hover:bg-[var(--surface-2)]",
            !tool.available && "opacity-55"
          )}
        >
          <div className="flex items-center justify-between gap-2">
            <span className="font-mono text-xs">{tool.name}</span>
            <span className={cn("text-xs", statusClass(tool.status))}>{tool.status}</span>
          </div>
          <div className="mt-1 line-clamp-2 text-xs text-[var(--text-faint)]">
            {tool.description}
          </div>
        </button>
      ))}
    </div>
  );
}

/** The skill-toggle grid, reused inside SkillsDialog. Toggles persist immediately. */
function SkillsGrid({
  doc,
  agent,
  skills,
  enabled,
  loading,
  onToggle,
}: {
  doc: AgentConfigDocument;
  agent: AgentProfile;
  skills: CapabilityConfig[];
  enabled: Set<string>;
  loading: boolean;
  onToggle: (skillId: string) => void;
}) {
  return (
    <>
      <div className="grid gap-2 md:grid-cols-2">
        {skills.map((skill) => (
          <button
            key={skill.id}
            type="button"
            disabled={loading || skill.status !== "ready"}
            onClick={() => onToggle(skill.id)}
            className={cn(
              "rounded-[var(--radius-sm)] border px-3 py-2 text-left text-sm",
              enabled.has(skill.id)
                ? "border-[var(--accent)] bg-[var(--surface-2)] text-[var(--text)]"
                : "border-[var(--border)] text-[var(--text-muted)] hover:bg-[var(--surface-2)]",
              skill.status !== "ready" && "opacity-55"
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span>{skill.name}</span>
              <span className={cn("text-xs", statusClass(skill.status))}>
                {statusLabel(skill)}
              </span>
            </div>
            <div className="mt-1 text-xs text-[var(--text-faint)]">
              {skill.dependencies.length ? `Requires ${skill.dependencies.join(", ")}` : "No tool dependency"}
            </div>
          </button>
        ))}
      </div>
      {skills.length === 0 && (
        <p className="text-xs text-[var(--text-faint)]">No skills installed yet.</p>
      )}
      {skillSummary(doc, agent).length > 0 && (
        <div className="mt-3 text-xs text-[var(--text-faint)]">
          Enabled: {skillSummary(doc, agent).map((skill) => skill.name).join(", ")}
        </div>
      )}
    </>
  );
}

/** Control Room → Default Persona. Edits the deployment-wide template
 * (doc.default_instructions) that SEEDS a new agent's Instructions at creation.
 * It is a snapshot copy — editing it never changes existing agents, and it is
 * never merged into them at runtime. Commits on blur (one PUT). */
function OrgPersonaPanel({
  doc,
  onSave,
}: {
  doc: AgentConfigDocument;
  onSave: (doc: AgentConfigDocument, message?: string) => Promise<void>;
}) {
  const { t } = useI18n();
  const [text, setText] = useState<string>(doc.default_instructions ?? "");
  const dirty = text !== (doc.default_instructions ?? "");

  const commit = () => {
    if (dirty) {
      void onSave({ ...doc, default_instructions: text });
    }
  };

  return (
    <>
      <Header
        title={t("settings.navDefaultPersona")}
        body={t("settings.defaultPersonaBody")}
      />
      <div className={cn(CARD, "p-0")}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={18}
          placeholder={t("settings.defaultPersonaPlaceholder")}
          className="block w-full resize-y rounded-t-[var(--radius)] border-0 border-b border-[var(--border)] bg-[var(--bg)] px-3 py-2 font-mono text-xs leading-6 outline-none focus:shadow-[var(--shadow-focus)]"
        />
        <div className="flex min-h-12 items-center justify-between gap-3 px-3 py-2">
          <span className="text-xs text-[var(--text-faint)]">
            {dirty ? t("settings.unsavedChanges") : t("settings.saved")}
          </span>
          <button
            type="button"
            disabled={!dirty}
            onClick={commit}
            className={BTN_PRIMARY}
          >
            {t("common.save")}
          </button>
        </div>
      </div>
    </>
  );
}

/** Per-agent code-repository manifest editor. Local state (keyed on agent id by
 * the parent, so it resets on switch) avoids a whole-document PUT per keystroke —
 * it commits on blur and after an AI generation. The "generate" button drives the
 * backend to explore /mnt/code with the LLM and returns Markdown for review. */
function CodeManifestSection({
  doc,
  agent,
  loading,
  onSave,
}: {
  doc: AgentConfigDocument;
  agent: AgentProfile;
  loading: boolean;
  onSave: (doc: AgentConfigDocument, message?: string) => Promise<void>;
}) {
  const { t } = useI18n();
  const [value, setValue] = useState(agent.code_manifest ?? "");
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");

  const commit = (next: string) => {
    if (next !== (agent.code_manifest ?? "")) {
      void onSave(applyAgentPatch(doc, agent.id, { code_manifest: next }));
    }
  };

  const onGenerate = async () => {
    setGenerating(true);
    setError("");
    try {
      const { manifest } = await generateCodeManifest(agent.id);
      setValue(manifest);
      commit(manifest);
      toast.success(t("settings.manifestGenerated"));
    } catch (err) {
      setError(err instanceof Error ? err.message : t("settings.generateFailed"));
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className={CARD}>
      <div className="mb-1 flex items-center gap-2">
        <GitBranch className="h-4 w-4 text-[var(--text-muted)]" />
        <h3 className="text-sm font-semibold">{t("settings.codeManifest")}</h3>
        <button
          type="button"
          onClick={() => void onGenerate()}
          disabled={generating || loading}
          title={t("settings.manifestGenTitle")}
          className="ml-auto inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] border border-[var(--border)] px-2.5 py-1 text-xs hover:bg-[var(--surface-2)] disabled:opacity-50"
        >
          {generating ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Sparkles className="h-3.5 w-3.5" />
          )}
          {generating ? t("settings.generating") : t("settings.aiGenerate")}
        </button>
      </div>
      <p className="mb-2 text-xs text-[var(--text-muted)]">
        {t("settings.manifestDescA")}<code>/mnt/code</code>{t("settings.manifestDescB")}
      </p>
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onBlur={() => commit(value)}
        rows={8}
        placeholder={t("settings.manifestPlaceholder")}
        className="w-full resize-y rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 font-mono text-xs"
      />
      {error && (
        <p className="mt-2 text-xs text-[var(--warning,#d97706)]">{error}</p>
      )}
    </div>
  );
}

// Per-agent knowledge scoping. The checkboxes bind to agent.knowledge.kb_ids;
// leaving all unchecked means "every knowledge base the user can access" (the
// server intersects the chosen ids with the caller's permissions either way).
function KnowledgeSection({
  doc,
  agent,
  onSave,
  bare,
}: {
  doc: AgentConfigDocument;
  agent: AgentProfile;
  onSave: (doc: AgentConfigDocument, message?: string) => Promise<void>;
  /** Render just the description + checklist (no card/header) for use in a dialog. */
  bare?: boolean;
}) {
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [loadError, setLoadError] = useState(false);
  const initialRerank = agent.knowledge?.rerank ?? {
    enabled: false,
    model: "",
    candidate_pool_size: 50,
  };
  const [rerankModel, setRerankModel] = useState(initialRerank.model);
  const [poolText, setPoolText] = useState(String(initialRerank.candidate_pool_size));

  useEffect(() => {
    let alive = true;
    listKnowledgeBases()
      .then((list) => alive && setKbs(list))
      .catch(() => alive && setLoadError(true));
    return () => {
      alive = false;
    };
  }, []);

  const selected = new Set(agent.knowledge?.kb_ids ?? []);
  const scoped = selected.size > 0;
  const rerankModels = (doc.models.providers ?? []).flatMap((provider) =>
    (provider.models ?? [])
      .filter((model) => model.type === "rerank")
      .map((model) => `${provider.name}/${model.id}`)
  );

  const saveRerank = (model: string, rawPool: string) => {
    const parsed = Number(rawPool);
    const candidatePoolSize = Math.max(
      1,
      Math.min(200, Number.isFinite(parsed) ? Math.round(parsed) : 50)
    );
    setPoolText(String(candidatePoolSize));
    void onSave(
      applyAgentPatch(doc, agent.id, {
        knowledge: {
          kb_ids: [...selected],
          rerank: {
            enabled: Boolean(model),
            model,
            candidate_pool_size: candidatePoolSize,
          },
        },
      })
    );
  };

  const toggle = (kbId: string) => {
    const next = new Set(selected);
    if (next.has(kbId)) next.delete(kbId);
    else next.add(kbId);
    void onSave(
      applyAgentPatch(doc, agent.id, {
        knowledge: {
          kb_ids: [...next],
          rerank: {
            enabled: Boolean(rerankModel),
            model: rerankModel,
            candidate_pool_size: Number(poolText) || 50,
          },
        },
      })
    );
  };

  const inner = (
    <>
      <p className="mb-3 text-xs text-[var(--text-muted)]">
        {scoped
          ? "This agent defaults its knowledge search to the bases checked below."
          : "Nothing checked — this agent searches every knowledge base the user can access."}
      </p>
      {loadError ? (
        <p className="text-xs text-[var(--text-faint)]">Couldn’t load knowledge bases.</p>
      ) : kbs.length === 0 ? (
        <p className="text-xs text-[var(--text-faint)]">No knowledge bases yet.</p>
      ) : (
        <div className="space-y-1">
          {kbs.map((kb) => (
            <label
              key={kb.id}
              className="flex items-center gap-2 rounded-[var(--radius-sm)] px-2 py-1.5 text-sm hover:bg-[var(--surface-2)]"
            >
              <input
                type="checkbox"
                checked={selected.has(kb.id)}
                onChange={() => toggle(kb.id)}
              />
              <span>{kb.name}</span>
              <span className="ml-auto text-xs text-[var(--text-faint)]">{kb.visibility}</span>
            </label>
          ))}
        </div>
      )}
      <div className="mt-4 grid gap-3 border-t border-[var(--border)] pt-4 sm:grid-cols-[minmax(0,1fr)_150px]">
        <label className="block text-xs font-medium text-[var(--text-muted)]">
          <span className="mb-1.5 block">Rerank model</span>
          <select
            aria-label="Rerank model"
            className={INPUT}
            value={rerankModel}
            onChange={(event) => {
              const model = event.target.value;
              setRerankModel(model);
              saveRerank(model, poolText);
            }}
          >
            <option value="">Disabled</option>
            {rerankModels.map((model) => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </label>
        <label className="block text-xs font-medium text-[var(--text-muted)]">
          <span className="mb-1.5 block">Candidate pool size</span>
          <input
            aria-label="Candidate pool size"
            className={cn(INPUT, "font-mono")}
            type="number"
            min={1}
            max={200}
            disabled={!rerankModel}
            value={poolText}
            onChange={(event) => setPoolText(event.target.value)}
            onBlur={() => saveRerank(rerankModel, poolText)}
          />
        </label>
      </div>
    </>
  );

  if (bare) return inner;

  return (
    <div className={CARD}>
      <div className="mb-1 flex items-center gap-2">
        <Database className="h-4 w-4 text-[var(--text-muted)]" />
        <h3 className="text-sm font-semibold">Knowledge</h3>
      </div>
      {inner}
    </div>
  );
}

/** Tools editor dialog: the per-agent tool grid + (when code browsing is on) the
 * code-repository manifest, which belongs with tools. Toggles persist immediately. */
function ToolsDialog({
  doc,
  agent,
  tools,
  enabled,
  loading,
  onToggle,
  onSave,
  onClose,
}: {
  doc: AgentConfigDocument;
  agent: AgentProfile;
  tools: ReturnType<typeof systemTools>;
  enabled: Set<string>;
  loading: boolean;
  onToggle: (toolId: string) => void;
  onSave: (doc: AgentConfigDocument, message?: string) => Promise<void>;
  onClose: () => void;
}) {
  return (
    <EditorDialog title={`Capabilities — ${agent.name}`} onClose={onClose} wide>
      <ToolsGrid tools={tools} enabled={enabled} loading={loading} onToggle={onToggle} />
      {enabled.has("code_sandbox") && (
        <div className="mt-4">
          <CodeManifestSection doc={doc} agent={agent} loading={loading} onSave={onSave} />
        </div>
      )}
    </EditorDialog>
  );
}

/** Skills editor dialog: the per-agent skill grid. Toggles persist immediately. */
function SkillsDialog({
  doc,
  agent,
  skills,
  enabled,
  loading,
  onToggle,
  onClose,
}: {
  doc: AgentConfigDocument;
  agent: AgentProfile;
  skills: CapabilityConfig[];
  enabled: Set<string>;
  loading: boolean;
  onToggle: (skillId: string) => void;
  onClose: () => void;
}) {
  return (
    <EditorDialog title={`Skills — ${agent.name}`} onClose={onClose} wide>
      <SkillsGrid
        doc={doc}
        agent={agent}
        skills={skills}
        enabled={enabled}
        loading={loading}
        onToggle={onToggle}
      />
    </EditorDialog>
  );
}

/** Knowledge scoping dialog: the per-agent KB checklist. Toggles persist immediately. */
function KnowledgeDialog({
  doc,
  agent,
  onSave,
  onClose,
}: {
  doc: AgentConfigDocument;
  agent: AgentProfile;
  onSave: (doc: AgentConfigDocument, message?: string) => Promise<void>;
  onClose: () => void;
}) {
  return (
    <EditorDialog title={`Knowledge Scope — ${agent.name}`} onClose={onClose}>
      <KnowledgeSection doc={doc} agent={agent} onSave={onSave} bare />
    </EditorDialog>
  );
}

/** A compact summary tile in the agent overview: icon + label, the current value,
 * and a "✎ 编辑" affordance that opens the matching editor dialog. */
/** A read-only overview tile: icon + title + optional badge, an explicit "编辑"
 * button (nothing is editable until it's clicked), and a preview body. */
function PreviewCard({
  icon,
  title,
  badge,
  editLabel,
  onEdit,
  children,
}: {
  icon: ReactNode;
  title: string;
  badge?: ReactNode;
  editLabel: string;
  onEdit: () => void;
  children: ReactNode;
}) {
  const { t } = useI18n();
  return (
    <div className={cn(CARD, "professional-card-hover p-[18px]")}>
      <div className="mb-3 flex items-center gap-2 text-[var(--text-muted)]">
        {icon}
        <h3 className="text-[15px] font-semibold text-[var(--text)]">{title}</h3>
        {badge}
        <button
          type="button"
          aria-label={editLabel}
          onClick={onEdit}
          className="focus-ring ml-auto inline-flex items-center gap-1 rounded-[var(--radius)] border border-transparent px-2 py-1 text-xs font-medium text-[var(--accent)] hover:border-[var(--border)] hover:bg-[var(--surface)]"
        >
          <Pencil className="h-3 w-3" />
          {t("common.edit")}
        </button>
      </div>
      {children}
    </div>
  );
}

/** Comma-joined preview of enabled names, truncated to a few with a "+N" tail. */
function previewNames(names: string[], empty: string): string {
  if (names.length === 0) return empty;
  const shown = names.slice(0, 4);
  const rest = names.length - shown.length;
  return shown.join(", ") + (rest > 0 ? ` +${rest}` : "");
}

function AgentsPanel({
  doc,
  agent,
  agents,
  setSelectedAgentId,
  tools,
  skills,
  loading,
  onSave,
  onToggleTool,
  onToggleSkill,
}: {
  doc: AgentConfigDocument;
  agent: AgentProfile;
  agents: AgentProfile[];
  setSelectedAgentId: (id: string) => void;
  tools: ReturnType<typeof systemTools>;
  skills: CapabilityConfig[];
  loading: boolean;
  onSave: (doc: AgentConfigDocument, message?: string) => Promise<void>;
  onToggleTool: (toolId: string) => void;
  onToggleSkill: (skillId: string) => void;
}) {
  const { t } = useI18n();
  const enabledTools = enabledToolIds(agent, tools);
  const enabledSkills = new Set(agent.skills.enabled);
  // The resolved deployment default, surfaced by the backend on the llm.default
  // provider (falls back to the authored catalog default). A blank agent.model
  // inherits this at request time.
  const llmProvider = doc.providers.find((p) => p.id === "llm.default");
  const defaultModel = String(
    (llmProvider?.settings as { default_model?: unknown } | undefined)?.default_model ??
      doc.models.default_model ??
      ""
  );
  // Chat models registered across the connection catalog, as `provider/id`.
  const chatModels: string[] = (doc.models.providers ?? []).flatMap((p) =>
    (p.models ?? [])
      .filter((m) => m.type === "chat" || m.type === undefined)
      .map((m) => `${p.name}/${m.id}`)
  );
  // Keep an already-chosen override selectable even if it's no longer in the catalog.
  const modelOptions =
    agent.model && !chatModels.includes(agent.model)
      ? [agent.model, ...chatModels]
      : chatModels;

  // Which focused editor dialog is open, and the two-step delete confirm. Both
  // reset when the selected agent changes.
  const [dialog, setDialog] = useState<
    null | "persona" | "tools" | "skills" | "knowledge"
  >(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [modelEditing, setModelEditing] = useState(false);
  useEffect(() => {
    setDialog(null);
    setConfirmDelete(false);
    setModelEditing(false);
  }, [agent.id]);

  // Knowledge bases are fetched once so the overview can preview scoped KB *names*
  // (not just a count). The dialog fetches its own copy for the live checklist.
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  useEffect(() => {
    let alive = true;
    listKnowledgeBases()
      .then((list) => alive && setKbs(list))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  const isDefault = doc.default_agent === agent.id;
  const canDelete = agents.length > 1;
  const persona = (agent.instructions ?? "").trim();

  // Enabled names, for the read-only preview lines on each card.
  const kbCount = agent.knowledge?.kb_ids?.length ?? 0;
  const scopedKb = kbCount > 0;
  const kbIdSet = new Set(agent.knowledge?.kb_ids ?? []);
  const kbNames = kbs.filter((k) => kbIdSet.has(k.id)).map((k) => k.name);
  const toolNames = tools.filter((tl) => enabledTools.has(tl.id)).map((tl) => tl.name);
  const skillNames = skills.filter((s) => enabledSkills.has(s.id)).map((s) => s.name);

  // Deletion is a whole-doc PUT with one fewer agent. If the removed agent was the
  // deployment default, hand default to the first survivor; then reselect it.
  const deleteAgent = () => {
    const remaining = doc.agents.filter((a) => a.id !== agent.id);
    if (remaining.length === 0) return;
    const nextDefault =
      doc.default_agent === agent.id ? remaining[0].id : doc.default_agent;
    void onSave(
      { ...doc, agents: remaining, default_agent: nextDefault },
      "Agent deleted"
    );
    setConfirmDelete(false);
    setSelectedAgentId(remaining[0].id);
  };
  const makeDefault = () =>
    void onSave({ ...doc, default_agent: agent.id }, "Default agent set");
  const createAgent = () => {
    const existing = new Set(agents.map((a) => a.id));
    let n = agents.length + 1;
    let id = `agent-${n}`;
    while (existing.has(id)) id = `agent-${++n}`;
    const created = newAgentProfile(doc, id, `New agent ${n}`);
    void onSave(
      { ...doc, agents: [...doc.agents, created] },
      "Agent created"
    );
    setSelectedAgentId(id);
  };
  const resolvedModel = agent.model || defaultModel || "deployment default";

  return (
    <>
      <div className={cn(CARD, "mb-5 p-[18px]")}>
        <div className="flex flex-wrap items-start gap-3">
          <span className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] text-[var(--text-muted)]">
            <UserRound className="h-4 w-4" />
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="truncate text-lg font-semibold tracking-tight text-[var(--text)]">
                {agent.name}
              </h2>
              {isDefault && (
                <span className="rounded-full bg-[var(--surface-2)] px-2 py-0.5 text-[10px] font-medium text-[var(--text-muted)]">
                  {t("settings.defaultAgent")}
                </span>
              )}
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-[var(--text-muted)]">
              <span className="font-medium text-[var(--text-faint)]">Model</span>
              {modelEditing ? (
                <select
                  aria-label="Model"
                  autoFocus
                  value={agent.model}
                  onBlur={() => setModelEditing(false)}
                  onChange={(event) => {
                    void onSave(applyAgentPatch(doc, agent.id, { model: event.target.value }));
                    setModelEditing(false);
                  }}
                  className="h-8 min-w-[260px] rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-elevated)] px-2.5 font-mono text-xs text-[var(--text)] outline-none focus:border-[var(--accent)] focus:shadow-[var(--shadow-focus)]"
                >
                  <option value="">
                    {defaultModel ? `Inherit (${defaultModel})` : "Inherit deployment default"}
                  </option>
                  {modelOptions.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              ) : (
                <>
                  <span className="font-mono text-[var(--text-muted)]">{resolvedModel}</span>
                  {!agent.model && defaultModel && (
                    <span className="rounded-full bg-[var(--surface)] px-1.5 py-0.5 text-[10px]">
                      {t("settings.inheritsDefault")}
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={() => setModelEditing(true)}
                    className="focus-ring inline-flex h-6 items-center gap-1 rounded-[var(--radius-sm)] px-1.5 text-[11px] font-medium text-[var(--text-muted)] hover:bg-[var(--surface)] hover:text-[var(--text)]"
                  >
                    <Pencil className="h-3 w-3" />
                    {t("common.edit")}
                  </button>
                </>
              )}
            </div>
          </div>
          <div className="ml-auto flex flex-wrap justify-end gap-2">
            {!isDefault && (
              <button type="button" onClick={makeDefault} className={BTN_GHOST}>
                {t("settings.setDefault")}
              </button>
            )}
            <button type="button" onClick={createAgent} className={BTN_GHOST}>
              <Plus className="h-4 w-4" />
              {t("settings.newAgent")}
            </button>
            {confirmDelete ? (
              <>
                <button type="button" onClick={deleteAgent} className={BTN_DANGER}>
                  {t("settings.confirmDelete")}
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmDelete(false)}
                  className={BTN_GHOST}
                >
                  {t("common.cancel")}
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={() => setConfirmDelete(true)}
                disabled={!canDelete}
                title={canDelete ? t("settings.deleteThisAgent") : t("settings.keepOneAgent")}
                className={BTN_GHOST}
              >
                <Trash2 className="h-4 w-4" />
                {t("common.delete")}
              </button>
            )}
          </div>
        </div>

        {agents.length > 1 && (
          <div className="mt-4 flex flex-wrap gap-2 border-t border-[var(--border)] pt-3">
            {agents.map((item) => {
              const selected = item.id === agent.id;
              const itemIsDefault = doc.default_agent === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  aria-pressed={selected}
                  onClick={() => setSelectedAgentId(item.id)}
                  className={cn(
                    "focus-ring inline-flex h-8 max-w-[220px] items-center gap-1.5 rounded-[var(--radius)] border px-2.5 text-xs transition-colors",
                    selected
                      ? "border-[var(--border-strong)] bg-[var(--bg-elevated)] font-semibold text-[var(--text)] shadow-[var(--shadow-sm)]"
                      : "border-[var(--border)] bg-[var(--surface)] text-[var(--text-muted)] hover:border-[var(--border-strong)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text)]"
                  )}
                >
                  <span className="truncate">{item.name}</span>
                  {itemIsDefault && (
                    <span className="shrink-0 rounded-full bg-[var(--surface-2)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-faint)]">
                      {t("settings.defaultBadge")}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        )}
      </div>

      <div className="space-y-4">
        {/* Persona — first ~200 chars of the system prompt. */}
        <PreviewCard
          icon={<UserRound className="h-4 w-4" />}
          title="Persona"
          editLabel={t("settings.editPersona")}
          onEdit={() => setDialog("persona")}
        >
          <p className="line-clamp-3 whitespace-pre-wrap font-mono text-xs leading-6 text-[var(--text-muted)]">
            {persona
              ? persona.slice(0, 200) + (persona.length > 200 ? "…" : "")
              : t("settings.personaEmpty")}
          </p>
        </PreviewCard>

        {/* Capabilities — tools / skills / knowledge, previewing example names. */}
        <div className="grid gap-4 sm:grid-cols-3">
          <PreviewCard
            icon={<Wrench className="h-4 w-4" />}
            title={t("settings.capabilities")}
            badge={
              <span className="rounded-full bg-[var(--surface-2)] px-2 py-0.5 text-[10px] font-medium text-[var(--text-muted)]">
                {enabledTools.size}/{tools.length}
              </span>
            }
            editLabel={t("settings.editCapabilities")}
            onEdit={() => setDialog("tools")}
          >
            <p className="line-clamp-2 text-xs text-[var(--text-muted)]">
              {previewNames(toolNames, t("settings.noCapabilitiesEnabled"))}
            </p>
          </PreviewCard>
          <PreviewCard
            icon={<Sparkles className="h-4 w-4" />}
            title={t("settings.skills")}
            badge={
              <span className="rounded-full bg-[var(--surface-2)] px-2 py-0.5 text-[10px] font-medium text-[var(--text-muted)]">
                {enabledSkills.size}/{skills.length}
              </span>
            }
            editLabel={t("settings.editSkills")}
            onEdit={() => setDialog("skills")}
          >
            <p className="line-clamp-2 text-xs text-[var(--text-muted)]">
              {previewNames(skillNames, skills.length ? t("settings.noSkillsEnabled") : t("settings.noSkillsInstalled"))}
            </p>
          </PreviewCard>
          <PreviewCard
            icon={<Database className="h-4 w-4" />}
            title={t("settings.knowledgeScope")}
            badge={
              <span className="rounded-full bg-[var(--surface-2)] px-2 py-0.5 text-[10px] font-medium text-[var(--text-muted)]">
                {scopedKb ? kbCount : t("settings.all")}
              </span>
            }
            editLabel={t("settings.editKnowledge")}
            onEdit={() => setDialog("knowledge")}
          >
            <p className="line-clamp-2 text-xs text-[var(--text-muted)]">
              {scopedKb
                ? kbNames.length
                  ? previewNames(kbNames, "")
                  : t("settings.kbSelected", { count: kbCount })
                : t("settings.allKb")}
            </p>
          </PreviewCard>
        </div>
      </div>

      {dialog === "persona" && (
        <PersonaDialog
          doc={doc}
          agent={agent}
          onClose={() => setDialog(null)}
          onSave={onSave}
        />
      )}
      {dialog === "tools" && (
        <ToolsDialog
          doc={doc}
          agent={agent}
          tools={tools}
          enabled={enabledTools}
          loading={loading}
          onToggle={onToggleTool}
          onSave={onSave}
          onClose={() => setDialog(null)}
        />
      )}
      {dialog === "skills" && (
        <SkillsDialog
          doc={doc}
          agent={agent}
          skills={skills}
          enabled={enabledSkills}
          loading={loading}
          onToggle={onToggleSkill}
          onClose={() => setDialog(null)}
        />
      )}
      {dialog === "knowledge" && (
        <KnowledgeDialog
          doc={doc}
          agent={agent}
          onSave={onSave}
          onClose={() => setDialog(null)}
        />
      )}
    </>
  );
}

function ToolsPanel({
  tools,
  loading,
  onConfigureSearch,
  onConfigureSandbox,
  onConfigureAliyun,
  onConfigureVectorDB,
  onPatchCapability,
}: {
  tools: CapabilityConfig[];
  loading: boolean;
  onConfigureSearch: () => void;
  onConfigureSandbox: () => void;
  onConfigureAliyun: () => void;
  onConfigureVectorDB: () => void;
  onPatchCapability: (id: string, patch: Partial<CapabilityConfig>) => Promise<void>;
}) {
  const { t } = useI18n();
  return (
    <>
      <Header
        title={t("settings.capabilities")}
        body={t("settings.capabilitiesBody")}
      />
      <div className="grid gap-3 lg:grid-cols-3">
        {tools.map((cap) => (
          <div
            key={cap.id}
            className={CARD}
          >
            <div className="mb-3 flex items-start gap-3">
              <div className="rounded-[var(--radius-sm)] bg-[var(--surface-2)] p-2 text-[var(--text-muted)]">
                {iconFor(cap.id)}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-2">
                  <h4 className="text-sm font-semibold">{cap.name}</h4>
                  <span className={cn("text-xs", statusClass(cap.status))}>
                    {statusLabel(cap)}
                  </span>
                </div>
                <p className="mt-1 text-xs leading-5 text-[var(--text-muted)]">
                  {cap.description}
                </p>
              </div>
            </div>
            {cap.error && (
              <div className="mb-3 flex gap-2 rounded-[var(--radius-sm)] bg-[var(--warning)]/10 px-2 py-1.5 text-xs text-[var(--warning)]">
                <CircleAlert className="h-3.5 w-3.5 shrink-0" />
                {cap.error}
              </div>
            )}
            <div className="flex gap-2">
              <button
                type="button"
                disabled={loading}
                onClick={() =>
                  void onPatchCapability(cap.id, {
                    enabled: !cap.enabled,
                    permission: cap.permission === "admin"
                      ? "admin"
                      : cap.enabled ? "disabled" : "auto",
                  })
                }
                className={BTN_GHOST}
              >
                {cap.enabled ? "Disable" : "Enable"}
              </button>
              {cap.id === "search" && (
                <button
                  type="button"
                  onClick={onConfigureSearch}
                  className={BTN_GHOST}
                >
                  Configure
                </button>
              )}
              {cap.id === "sandbox" && (
                <button
                  type="button"
                  onClick={onConfigureSandbox}
                  className={BTN_GHOST}
                >
                  Configure
                </button>
              )}
              {cap.id === "knowledge" && (
                <button
                  type="button"
                  onClick={onConfigureVectorDB}
                  className={BTN_GHOST}
                >
                  Vector DB
                </button>
              )}
              {cap.id === "aliyun_pai" && (
                <button
                  type="button"
                  onClick={onConfigureAliyun}
                  className={BTN_GHOST}
                >
                  Authorize
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </>
  );
}

function SkillsPanel({
  skills,
  loading,
  onInstall,
  onPatchCapability,
}: {
  skills: CapabilityConfig[];
  loading: boolean;
  onInstall: () => void;
  onPatchCapability: (id: string, patch: Partial<CapabilityConfig>) => Promise<void>;
}) {
  const { t } = useI18n();
  return (
    <>
      <div className="mb-5 flex items-start justify-between gap-3">
        <Header
          title={t("settings.skills")}
          body={t("settings.skillsBody")}
        />
        <button
          type="button"
          onClick={onInstall}
          className={cn(BTN_PRIMARY, "mt-1 shrink-0")}
        >
          <Upload className="h-3.5 w-3.5" />
          {t("settings.installSkill")}
        </button>
      </div>
      <div className="grid gap-2 md:grid-cols-2">
        {skills.map((skill) => (
          <div
            key={skill.id}
            className="flex items-start justify-between gap-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-3"
          >
            <div>
              <div className="text-sm font-medium">{skill.name}</div>
              <div className="mt-1 text-xs leading-5 text-[var(--text-muted)]">
                {skill.description}
              </div>
              <div className="mt-2 text-xs text-[var(--text-faint)]">
                {skill.dependencies.length
                  ? t("settings.skillRequires", { list: skill.dependencies.join(", ") })
                  : t("settings.skillNoDependency")}
              </div>
              {skillMeta(skill) && (
                <div className="mt-1 text-xs text-[var(--text-faint)]">
                  {skillMeta(skill)}
                </div>
              )}
            </div>
            <button
              type="button"
              disabled={loading}
              onClick={() => void onPatchCapability(skill.id, { enabled: !skill.enabled })}
              className={cn(
                "shrink-0 rounded-[var(--radius-sm)] px-3 py-1.5 text-xs",
                skill.enabled
                  ? "bg-[var(--surface-2)] text-[var(--text)]"
                  : "border border-[var(--border)] text-[var(--text-muted)]"
              )}
            >
              {skill.enabled ? t("settings.skillInstalled") : t("common.disabled")}
            </button>
          </div>
        ))}
      </div>
    </>
  );
}

type SkillInstallMode = "zip_upload" | "url" | "git";

function SkillInstallDialog({
  agents,
  loading,
  onClose,
  onUpload,
  onInstall,
}: {
  agents: AgentProfile[];
  loading: boolean;
  onClose: () => void;
  onUpload: (file: File) => Promise<{ upload_id: string; filename?: string; size: number }>;
  onInstall: (payload: {
    source: {
      type: "zip_upload" | "url" | "git";
      upload_id?: string;
      url?: string;
      checksum?: string;
      ref?: string;
      path?: string;
    };
    enable_for_agent?: string;
    enable_after_build?: boolean;
    overwrite?: boolean;
  }) => Promise<void>;
}) {
  const [mode, setMode] = useState<SkillInstallMode>("zip_upload");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [checksum, setChecksum] = useState("");
  const [ref, setRef] = useState("");
  const [path, setPath] = useState("");
  const [agentId, setAgentId] = useState("");
  const [overwrite, setOverwrite] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const submit = async () => {
    setSubmitting(true);
    try {
      if (mode === "zip_upload") {
        if (!file) throw new Error("Choose a skill zip first");
        const upload = await onUpload(file);
        await onInstall({
          source: { type: "zip_upload", upload_id: upload.upload_id },
          enable_for_agent: agentId || undefined,
          overwrite,
        });
      } else if (mode === "url") {
        await onInstall({
          source: {
            type: "url",
            url: url.trim(),
            checksum: checksum.trim() || undefined,
          },
          enable_for_agent: agentId || undefined,
          overwrite,
        });
      } else {
        await onInstall({
          source: {
            type: "git",
            url: url.trim(),
            ref: ref.trim() || undefined,
            path: path.trim() || undefined,
          },
          enable_for_agent: agentId || undefined,
          overwrite,
        });
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Skill install failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/25 px-4">
      <div className="w-full max-w-lg rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-elevated)] shadow-xl">
        <div className="flex items-center gap-3 border-b border-[var(--border)] px-4 py-3">
          <div className="rounded-[var(--radius-sm)] bg-[var(--surface-2)] p-2 text-[var(--text-muted)]">
            {mode === "git" ? <GitBranch className="h-4 w-4" /> : <Upload className="h-4 w-4" />}
          </div>
          <div>
            <div className="text-sm font-semibold">Install Skill</div>
            <div className="text-xs text-[var(--text-muted)]">Admin-only install from ZIP, URL, or Git.</div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="ml-auto rounded-[var(--radius-sm)] p-1 text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-4 p-4">
          <div className="grid grid-cols-3 gap-1 rounded-[var(--radius-sm)] bg-[var(--surface-2)] p-1">
            {[
              ["zip_upload", "ZIP"],
              ["url", "URL"],
              ["git", "Git"],
            ].map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setMode(id as SkillInstallMode)}
                className={cn(
                  "rounded-[var(--radius-sm)] px-3 py-1.5 text-xs",
                  mode === id
                    ? "bg-[var(--surface)] text-[var(--text)] shadow-sm"
                    : "text-[var(--text-muted)] hover:text-[var(--text)]"
                )}
              >
                {label}
              </button>
            ))}
          </div>

          {mode === "zip_upload" && (
            <label className="block text-sm">
              <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Skill ZIP</span>
              <input
                type="file"
                accept=".zip,application/zip"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                className={INPUT}
              />
            </label>
          )}

          {mode === "url" && (
            <>
              <TextInput label="ZIP URL" value={url} onChange={setUrl} placeholder="https://example.com/skill.zip" />
              <TextInput label="SHA-256 checksum" value={checksum} onChange={setChecksum} placeholder="optional" />
            </>
          )}

          {mode === "git" && (
            <>
              <TextInput label="Git URL" value={url} onChange={setUrl} placeholder="https://github.com/org/repo.git" />
              <div className="grid gap-3 md:grid-cols-2">
                <TextInput label="Ref" value={ref} onChange={setRef} placeholder="main" />
                <TextInput label="Path" value={path} onChange={setPath} placeholder="skills/report-writer" />
              </div>
            </>
          )}

          <div className="grid gap-3 md:grid-cols-2">
            <label className="block text-sm">
              <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Enable For Agent</span>
              <select
                value={agentId}
                onChange={(event) => setAgentId(event.target.value)}
                className={INPUT}
              >
                <option value="">Do not enable</option>
                {agents.map((agent) => (
                  <option key={agent.id} value={agent.id}>{agent.name}</option>
                ))}
              </select>
            </label>
            <label className="flex items-end gap-2 pb-2 text-sm text-[var(--text-muted)]">
              <input
                type="checkbox"
                checked={overwrite}
                onChange={(event) => setOverwrite(event.target.checked)}
              />
              Overwrite existing skill
            </label>
          </div>
        </div>

        <div className="flex justify-end gap-2 border-t border-[var(--border)] px-4 py-3">
          <button
            type="button"
            onClick={onClose}
            className={BTN_GHOST}
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={loading || submitting}
            onClick={() => void submit()}
            className={BTN_PRIMARY}
          >
            {submitting ? "Installing" : "Install"}
          </button>
        </div>
      </div>
    </div>
  );
}

function TextInput({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">{label}</span>
      <input
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
        className={INPUT}
      />
    </label>
  );
}

function YamlPanel({
  yamlText,
  setYamlText,
  loading,
  onLoad,
  onSave,
}: {
  yamlText: string;
  setYamlText: (text: string) => void;
  loading: boolean;
  onLoad: () => Promise<void>;
  onSave: () => Promise<void>;
}) {
  const { t } = useI18n();
  return (
    <div className="flex min-h-[calc(100vh_-_var(--header-h)_-_48px)] flex-col">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <div className="min-w-0 flex-1">
          <h2 className="text-lg font-semibold tracking-tight">{t("settings.navYaml")}</h2>
          <p className="mt-1 text-xs leading-5 text-[var(--text-muted)]">
            {t("settings.yamlBody")}
          </p>
        </div>
        <div className="flex shrink-0 gap-2">
          <button
            type="button"
            onClick={() => void onLoad()}
            className={BTN_GHOST}
          >
            {t("settings.reloadYaml")}
          </button>
          <button
            type="button"
            disabled={loading}
            onClick={() => void onSave()}
            className={BTN_PRIMARY}
          >
            {t("settings.saveYaml")}
          </button>
        </div>
      </div>
      <textarea
        value={yamlText}
        onChange={(event) => setYamlText(event.target.value)}
        spellCheck={false}
        className="min-h-[680px] flex-1 resize-none rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4 font-mono text-[13px] leading-6 text-[var(--text)] outline-none focus:border-[var(--border-strong)] focus:shadow-[var(--shadow-focus)]"
      />
    </div>
  );
}

function SearchConfigDialog({
  doc,
  loading,
  testing,
  output,
  onClose,
  onSave,
  onTest,
}: {
  doc: AgentConfigDocument;
  loading: boolean;
  testing: boolean;
  output: string;
  onClose: () => void;
  onSave: (doc: AgentConfigDocument) => Promise<void>;
  onTest: () => void;
}) {
  const provider = doc.providers.find((p) => p.id === "search.default");
  const search = doc.capabilities.find((cap) => cap.id === "search");
  const settings = provider?.settings ?? {};
  const [providerName, setProviderName] = useState(String(settings.provider ?? "tavily"));
  const [apiKey, setApiKey] = useState("");
  const [apiKeyEnv, setApiKeyEnv] = useState(String(settings.api_key_env ?? ""));
  const [endpoint, setEndpoint] = useState(String(settings.endpoint ?? ""));
  const [maxResults, setMaxResults] = useState(String(settings.max_results ?? 5));

  const saveSearch = async () => {
    const next: AgentConfigDocument = {
      ...doc,
      providers: doc.providers.map((p) =>
        p.id === "search.default"
          ? {
              ...p,
              name: providerName,
              settings: {
                ...p.settings,
                provider: providerName,
                endpoint,
                api_key_env: apiKeyEnv,
                max_results: Number(maxResults) || 5,
                ...(apiKey ? { api_key: apiKey } : {}),
              },
            }
          : p
      ),
      capabilities: doc.capabilities.map((cap) =>
        cap.id === "search"
          ? { ...cap, enabled: true, permission: "auto", status: "ready" }
          : cap
      ),
    };
    await onSave(next);
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/30 p-4">
      <div className="w-full max-w-xl rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-elevated)] shadow-xl">
        <div className="flex h-11 items-center border-b border-[var(--border)] px-4">
          <div className="text-sm font-semibold">Configure Search</div>
          <button
            type="button"
            aria-label="Close search configuration"
            onClick={onClose}
            className="ml-auto rounded-[var(--radius-sm)] p-1 text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="space-y-4 p-4">
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Provider</span>
            <select
              value={providerName}
              onChange={(event) => setProviderName(event.target.value)}
              className={INPUT}
            >
              <option value="tavily">Tavily</option>
              <option value="brave">Brave Search</option>
            </select>
          </label>
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">API Key</span>
            <input
              value={apiKey}
              type="password"
              placeholder={provider?.secret_configured ? "Already configured" : "Paste key or use env below"}
              onChange={(event) => setApiKey(event.target.value)}
              className={INPUT}
            />
          </label>
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">API Key Env</span>
            <input
              value={apiKeyEnv}
              placeholder={providerName === "tavily" ? "TAVILY_API_KEY" : "BRAVE_SEARCH_API_KEY"}
              onChange={(event) => setApiKeyEnv(event.target.value)}
              className={INPUT}
            />
            <span className="mt-1 block text-xs text-[var(--text-faint)]">
              Stored as an environment-variable name — the secret value stays on the server.
            </span>
          </label>
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Endpoint</span>
            <input
              value={endpoint}
              placeholder={providerName === "tavily" ? "https://api.tavily.com/search" : "https://api.search.brave.com/res/v1/web/search"}
              onChange={(event) => setEndpoint(event.target.value)}
              className={INPUT}
            />
          </label>
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Max Results</span>
            <input
              value={maxResults}
              type="number"
              min={1}
              max={20}
              onChange={(event) => setMaxResults(event.target.value)}
              className={INPUT}
            />
          </label>
          <div className="rounded-[var(--radius-sm)] bg-[var(--surface-2)] px-3 py-2 text-xs text-[var(--text-muted)]">
            Permission is currently saved as auto because tool approval is not wired yet.
          </div>
          {output && (
            <pre className="max-h-48 overflow-auto rounded-[var(--radius-sm)] bg-[var(--surface)] p-3 text-xs text-[var(--text-muted)] whitespace-pre-wrap">
              {output}
            </pre>
          )}
        </div>
        <div className="flex justify-end gap-2 border-t border-[var(--border)] px-4 py-3">
          <button
            type="button"
            onClick={onTest}
            disabled={testing || !search?.enabled}
            className={BTN_GHOST}
          >
            {testing ? "Testing..." : "Test search"}
          </button>
          <button
            type="button"
            onClick={onClose}
            className={BTN_GHOST}
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={loading}
            onClick={saveSearch}
            className={BTN_PRIMARY}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

function VectorDBConfigDialog({
  doc,
  loading,
  onClose,
  onSave,
}: {
  doc: AgentConfigDocument;
  loading: boolean;
  onClose: () => void;
  onSave: (doc: AgentConfigDocument) => Promise<void>;
}) {
  const { t } = useI18n();
  const vdb = doc.knowledgebase.vectordb;
  const [url, setUrl] = useState(vdb.url);
  const [indexPrefix, setIndexPrefix] = useState(vdb.index_prefix || "kb");
  const [apiKey, setApiKey] = useState("");
  const [apiKeyEnv, setApiKeyEnv] = useState(vdb.api_key_env);
  const [username, setUsername] = useState(vdb.username);
  const [password, setPassword] = useState("");
  const [passwordEnv, setPasswordEnv] = useState(vdb.password_env);
  const [verifyCerts, setVerifyCerts] = useState(vdb.verify_certs);
  const [timeout, setTimeoutValue] = useState(String(vdb.timeout ?? 30));

  const saveVectorDB = async () => {
    const next: AgentConfigDocument = {
      ...doc,
      knowledgebase: {
        ...doc.knowledgebase,
        vectordb: {
          // Spread first so untouched masked secrets ("********") flow through and
          // the backend restores them; only override api_key/password when typed.
          ...vdb,
          // Elasticsearch is the only user-facing engine; the local SQL scan
          // stays a backend-internal default/fallback and is never chosen here.
          engine: "elasticsearch",
          url,
          index_prefix: indexPrefix,
          api_key_env: apiKeyEnv,
          username,
          password_env: passwordEnv,
          verify_certs: verifyCerts,
          timeout: Number(timeout) || 30,
          ...(apiKey ? { api_key: apiKey } : {}),
          ...(password ? { password } : {}),
        },
      },
    };
    await onSave(next);
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/30 p-4">
      <div className="w-full max-w-xl rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-elevated)] shadow-xl">
        <div className="flex h-11 items-center border-b border-[var(--border)] px-4">
          <div className="text-sm font-semibold">Configure Vector Database</div>
          <button
            type="button"
            aria-label="Close vector database configuration"
            onClick={onClose}
            className="ml-auto rounded-[var(--radius-sm)] p-1 text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="max-h-[70vh] space-y-4 overflow-y-auto p-4">
          <div className="rounded-[var(--radius-sm)] bg-[var(--surface-2)] px-3 py-2 text-xs text-[var(--text-muted)]">
            {t("settings.vdbIntro")}
          </div>
              <label className="block text-sm">
                <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">URL</span>
                <input
                  value={url}
                  placeholder="https://es-host:9200"
                  onChange={(event) => setUrl(event.target.value)}
                  className={INPUT}
                />
              </label>
              <label className="block text-sm">
                <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Index Prefix</span>
                <input
                  value={indexPrefix}
                  placeholder="kb"
                  onChange={(event) => setIndexPrefix(event.target.value)}
                  className={INPUT}
                />
              </label>
              <div className="rounded-[var(--radius-sm)] bg-[var(--surface-2)] px-3 py-2 text-xs text-[var(--text-muted)]">
                {t("settings.vdbAuthHint")}
              </div>
              <label className="block text-sm">
                <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">API Key</span>
                <input
                  value={apiKey}
                  type="password"
                  placeholder={vdb.api_key ? "Already configured" : "Paste key or use env below"}
                  onChange={(event) => setApiKey(event.target.value)}
                  className={INPUT}
                />
              </label>
              <label className="block text-sm">
                <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">API Key Env</span>
                <input
                  value={apiKeyEnv}
                  placeholder="ELASTICSEARCH_API_KEY"
                  onChange={(event) => setApiKeyEnv(event.target.value)}
                  className={INPUT}
                />
                <span className="mt-1 block text-xs text-[var(--text-faint)]">
                  Stored as an environment-variable name — the secret value stays on the server.
                </span>
              </label>
              <label className="block text-sm">
                <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Username</span>
                <input
                  value={username}
                  placeholder="elastic"
                  onChange={(event) => setUsername(event.target.value)}
                  className={INPUT}
                />
              </label>
              <label className="block text-sm">
                <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Password</span>
                <input
                  value={password}
                  type="password"
                  placeholder={vdb.password ? "Already configured" : "Paste password or use env below"}
                  onChange={(event) => setPassword(event.target.value)}
                  className={INPUT}
                />
              </label>
              <label className="block text-sm">
                <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Password Env</span>
                <input
                  value={passwordEnv}
                  placeholder="ELASTICSEARCH_PASSWORD"
                  onChange={(event) => setPasswordEnv(event.target.value)}
                  className={INPUT}
                />
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={verifyCerts}
                  onChange={(event) => setVerifyCerts(event.target.checked)}
                  className="h-4 w-4"
                />
                <span className="text-xs font-medium text-[var(--text-muted)]">Verify TLS certificates</span>
              </label>
              <label className="block text-sm">
                <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Timeout (seconds)</span>
                <input
                  value={timeout}
                  type="number"
                  min={1}
                  onChange={(event) => setTimeoutValue(event.target.value)}
                  className={INPUT}
                />
              </label>

          <div className="rounded-[var(--radius-sm)] bg-[var(--surface-2)] px-3 py-2 text-xs text-[var(--text-muted)]">
            {t("settings.vdbGlobalNote")}
          </div>
        </div>
        <div className="flex justify-end gap-2 border-t border-[var(--border)] px-4 py-3">
          <button
            type="button"
            onClick={onClose}
            className={BTN_GHOST}
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={loading}
            onClick={saveVectorDB}
            className={BTN_PRIMARY}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

function SandboxConfigDialog({
  doc,
  loading,
  onClose,
  onSave,
}: {
  doc: AgentConfigDocument;
  loading: boolean;
  onClose: () => void;
  onSave: (doc: AgentConfigDocument) => Promise<void>;
}) {
  const provider = doc.providers.find((p) => p.id === "sandbox.default");
  const sandbox = doc.capabilities.find((cap) => cap.id === "sandbox");
  const settings = provider?.settings ?? {};
  const [endpoint, setEndpoint] = useState(String(settings.endpoint ?? ""));
  const [apiKey, setApiKey] = useState("");
  const [apiKeyEnv, setApiKeyEnv] = useState(String(settings.api_key_env ?? "AGENTRUN_SANDBOX_API_KEY"));
  const [accountId, setAccountId] = useState(String(settings.account_id ?? ""));
  const [accountIdEnv, setAccountIdEnv] = useState(String(settings.account_id_env ?? "AGENTRUN_ACCOUNT_ID"));
  const [templateName, setTemplateName] = useState(String(settings.template_name ?? ""));
  const [sessionIdle, setSessionIdle] = useState(String(settings.session_idle_seconds ?? 600));
  const [timeout, setTimeoutValue] = useState(String(settings.timeout_seconds ?? 30));
  const [error, setError] = useState("");

  const saveSandbox = async () => {
    setError("");
    // Backend contract: template_name + api_key (direct or env) + account_id
    // (direct or env) are required. The gateway endpoint is optional — when
    // omitted the backend auto-derives it from account_id + region.
    if (
      !templateName.trim() ||
      (!apiKey.trim() && !apiKeyEnv.trim()) ||
      (!accountId.trim() && !accountIdEnv.trim())
    ) {
      setError("Template name, API key, and account id are required");
      return;
    }

    const next: AgentConfigDocument = {
      ...doc,
      providers: doc.providers.map((p) =>
        p.id === "sandbox.default"
          ? {
              ...p,
              name: "AgentRun REST sandbox",
              settings: {
                ...p.settings,
                provider: "agentrun_rest",
                endpoint,
                api_key_env: apiKeyEnv,
                api_key_header: "X-API-Key",
                account_id_env: accountIdEnv,
                template_name: templateName,
                template_type: "CodeInterpreter",
                isolation_scope: "conversation",
                idle_timeout_seconds: Number(sessionIdle) || 600,
                session_idle_seconds: Number(sessionIdle) || 600,
                timeout_seconds: Math.max(1, Math.min(Number(timeout) || 30, 30)),
                cwd: "/home/user",
                create_path: "/sandboxes",
                execute_path: "/sandboxes/{sandbox_id}/contexts/execute",
                stop_path: "/sandboxes/{sandbox_id}/stop",
                oss_mount_config: p.settings.oss_mount_config ?? { mount_points: [] },
                nas_config: p.settings.nas_config ?? { mount_points: [] },
                ...(apiKey ? { api_key: apiKey } : {}),
                ...(accountId ? { account_id: accountId } : {}),
              },
            }
          : p
      ),
      capabilities: doc.capabilities.map((cap) =>
        cap.id === "sandbox"
          ? { ...cap, enabled: true, permission: "auto", status: "ready" }
          : cap
      ),
      agents: doc.agents.map((agent) => ({
        ...agent,
        tools: {
          include: Array.from(new Set([...agent.tools.include, ...toolBundle("code_sandbox")])),
          exclude: agent.tools.exclude.filter((name) => !toolAliases("code_sandbox").includes(name)),
        },
      })),
    };
    await onSave(next);
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/30 p-4">
      <div className="max-h-[92vh] w-full max-w-3xl overflow-hidden rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-elevated)] shadow-xl">
        <div className="flex h-11 items-center border-b border-[var(--border)] px-4">
          <div className="text-sm font-semibold">Configure Sandbox</div>
          <button
            type="button"
            aria-label="Close sandbox configuration"
            onClick={onClose}
            className="ml-auto rounded-[var(--radius-sm)] p-1 text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="max-h-[calc(92vh-96px)] space-y-4 overflow-y-auto p-4">
          <div className="grid gap-3 md:grid-cols-2">
            <Field label="Gateway endpoint">
              <input
                aria-label="Sandbox gateway endpoint"
                value={endpoint}
                placeholder="Auto-derived from account id if empty"
                onChange={(event) => setEndpoint(event.target.value)}
                className={INPUT}
              />
              {!endpoint.trim() && accountId.trim() && (
                <span className="mt-1 block text-xs text-[var(--text-muted)]">
                  Will use https://{accountId}.agentrun-data.cn-hangzhou.aliyuncs.com
                </span>
              )}
            </Field>
            <Field label="Template name">
              <input
                aria-label="Sandbox template name"
                value={templateName}
                placeholder="code-interpreter-template"
                onChange={(event) => setTemplateName(event.target.value)}
                className={INPUT}
              />
            </Field>
            <Field label="Gateway API key">
              <input
                aria-label="Sandbox gateway API key"
                value={apiKey}
                type="password"
                placeholder={provider?.secret_configured ? "Already configured" : "Optional if env is set"}
                onChange={(event) => setApiKey(event.target.value)}
                className={INPUT}
              />
            </Field>
            <Field label="Gateway API key env">
              <input
                aria-label="Sandbox gateway API key env"
                value={apiKeyEnv}
                placeholder="AGENTRUN_SANDBOX_API_KEY"
                onChange={(event) => setApiKeyEnv(event.target.value)}
                className={INPUT}
              />
              <span className="mt-1 block text-xs text-[var(--text-faint)]">
                Stored as an environment-variable name — the secret value stays on the server.
              </span>
            </Field>
            <Field label="Alibaba Cloud account ID">
              <input
                aria-label="Sandbox Alibaba Cloud account ID"
                value={accountId}
                placeholder="Optional if env is set"
                onChange={(event) => setAccountId(event.target.value)}
                className={INPUT}
              />
            </Field>
            <Field label="Account ID env">
              <input
                aria-label="Sandbox account ID env"
                value={accountIdEnv}
                placeholder="AGENTRUN_ACCOUNT_ID"
                onChange={(event) => setAccountIdEnv(event.target.value)}
                className={INPUT}
              />
            </Field>
            <Field label="Session idle seconds">
              <input
                aria-label="Sandbox session idle seconds"
                value={sessionIdle}
                type="number"
                min={30}
                onChange={(event) => setSessionIdle(event.target.value)}
                className={INPUT}
              />
            </Field>
            <Field label="Execution timeout seconds">
              <input
                aria-label="Sandbox execution timeout seconds"
                value={timeout}
                type="number"
                min={1}
                onChange={(event) => setTimeoutValue(event.target.value)}
                className={INPUT}
              />
            </Field>
          </div>

          <div className="rounded-[var(--radius-sm)] bg-[var(--surface-2)] px-3 py-2 text-xs text-[var(--text-muted)]">
            Sandboxes are isolated per conversation. Advanced mount and REST path settings stay in YAML.
          </div>
          {sandbox?.error && (
            <div className="rounded-[var(--radius-sm)] bg-[var(--warning)]/10 px-3 py-2 text-xs text-[var(--warning)]">
              {sandbox.error}
            </div>
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
            className={BTN_GHOST}
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={loading}
            onClick={saveSandbox}
            className={BTN_PRIMARY}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">{label}</span>
      {children}
    </label>
  );
}
