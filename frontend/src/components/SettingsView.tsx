import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  ArrowLeft,
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
import { useAgentConfigStore } from "../store/agentConfig";
import { useAliyunDialog } from "../store/aliyunDialog";
import { ConnectionsPanel } from "./ConnectionsPanel";
import { KnowledgeBasePanel } from "./KnowledgeBasePanel";
import { ThemeToggle } from "./ThemeToggle";
import { useI18n } from "../i18n";

type Tab = "agents" | "org-persona" | "tools" | "connections" | "knowledge" | "skills" | "yaml";

function statusClass(status: string) {
  if (status === "ready" || status === "healthy") return "text-[var(--success)]";
  if (status === "missing_config") return "text-[var(--warning)]";
  if (status === "error") return "text-[var(--danger)]";
  return "text-[var(--text-faint)]";
}

// Whether a Control Room tab is provisioned enough to use. Only tabs that carry
// a real provisioning signal return a verdict; the rest return null (no badge),
// so the nav shows a dot exactly where an operator has something to wire.
function tabReadiness(tab: Tab, doc: AgentConfigDocument): "ready" | "attention" | null {
  if (tab === "connections") {
    const llm = doc.providers.find((p) => p.id === "llm.default");
    return llm?.status === "healthy" ? "ready" : "attention";
  }
  if (tab === "knowledge") {
    const vdb = doc.knowledgebase.vectordb;
    // The local engine needs no external service; ES must report healthy.
    return vdb.engine === "local" || vdb.status === "healthy" ? "ready" : "attention";
  }
  if (tab === "tools") {
    // Flag when a switched-on core tool is missing its backing config.
    const broken = doc.capabilities.some(
      (c) => c.kind === "core_tool" && c.enabled && (c.status === "missing_config" || c.status === "error")
    );
    return broken ? "attention" : "ready";
  }
  return null;
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

function systemTools(doc: AgentConfigDocument) {
  const core = doc.capabilities
    // aliyun_pai is a deployment capability (governs sandbox credential
    // injection), not a callable tool an agent selects — skip it here.
    .filter((cap) => cap.kind === "core_tool" && cap.id !== "aliyun_pai")
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
}: {
  doc: AgentConfigDocument;
  onBack: () => void;
}) {
  const save = useAgentConfigStore((s) => s.save);
  const loadYaml = useAgentConfigStore((s) => s.loadYaml);
  const saveYaml = useAgentConfigStore((s) => s.saveYaml);
  const testSearch = useAgentConfigStore((s) => s.testSearch);
  const uploadSkillZip = useAgentConfigStore((s) => s.uploadSkillZip);
  const installSkill = useAgentConfigStore((s) => s.installSkill);
  const loading = useAgentConfigStore((s) => s.loading);
  const [tab, setTab] = useState<Tab>("agents");
  const [agentId, setAgentId] = useState(doc.default_agent || doc.agents[0]?.id || "main");
  const [searchOpen, setSearchOpen] = useState(false);
  const [sandboxOpen, setSandboxOpen] = useState(false);
  const [vectordbOpen, setVectordbOpen] = useState(false);
  const showAliyunDialog = useAliyunDialog((s) => s.show);
  const [skillInstallOpen, setSkillInstallOpen] = useState(false);
  const [yamlText, setYamlText] = useState("");
  const [testingSearch, setTestingSearch] = useState(false);
  const [searchOutput, setSearchOutput] = useState("");

  const agents = doc.agents.length ? doc.agents : [];
  const agent = agents.find((item) => item.id === agentId) ?? agents[0];
  const tools = useMemo(() => systemTools(doc), [doc]);
  const coreTools = doc.capabilities.filter((cap) => cap.kind === "core_tool");
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
    if (included.has(toolId)) {
      included.delete(toolId);
      excluded.add(toolId);
    } else {
      included.add(toolId);
      excluded.delete(toolId);
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
      setTab("yaml");
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

  // Two surfaces, split by who owns the decision. Agent Studio is where an
  // author shapes an individual agent (daily); Control Room is where an operator
  // wires the deployment-wide backends (rarely). The whole view stays admin-only.
  const tabGroups: Array<{ heading: string; items: Array<{ id: Tab; label: string }> }> = [
    {
      heading: "Agent Studio",
      items: [{ id: "agents", label: "Agents" }],
    },
    {
      heading: "Control Room",
      items: [
        { id: "org-persona", label: "Default Persona" },
        { id: "connections", label: "Connections" },
        { id: "tools", label: "Tools" },
        { id: "knowledge", label: "Knowledge Base" },
        { id: "skills", label: "Skills" },
        { id: "yaml", label: "YAML" },
      ],
    },
  ];

  return (
    <div className="flex h-full flex-col bg-[var(--bg)] text-[var(--text)]">
      <div className="flex h-12 items-center gap-2 border-b border-[var(--border)] px-3">
        <button
          type="button"
          aria-label="Back to chat"
          onClick={onBack}
          className="rounded-[var(--radius-sm)] p-1.5 text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
        >
          <ArrowLeft className="h-4 w-4" />
        </button>
        <h1 className="text-sm font-semibold">Settings</h1>
        <div className="flex-1" />
        <ThemeToggle />
      </div>

      <main className="mx-auto grid w-full max-w-6xl flex-1 grid-cols-[180px_1fr] gap-6 overflow-y-auto px-5 py-6">
        <aside className="space-y-5">
          {tabGroups.map((group) => (
            <div key={group.heading} className="space-y-1">
              <div className="px-3 pb-1 text-[11px] font-semibold uppercase tracking-wider text-[var(--text-faint)]">
                {group.heading}
              </div>
              {group.items.map((item) => {
                const readiness = tabReadiness(item.id, doc);
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={async () => {
                      if (item.id === "yaml" && !yamlText) await openYaml();
                      else setTab(item.id);
                    }}
                    className={cn(
                      "flex w-full items-center rounded-[var(--radius-sm)] px-3 py-2 text-left text-sm transition-colors",
                      tab === item.id
                        ? "bg-[var(--surface-2)] text-[var(--text)] font-medium"
                        : "text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
                    )}
                  >
                    {item.label}
                    {readiness && (
                      <span
                        // Decorative for the button's accessible name (which stays
                        // just the label); the title carries the readiness verdict.
                        aria-hidden="true"
                        title={
                          readiness === "ready"
                            ? `${item.label} — Ready`
                            : `${item.label} — Needs setup`
                        }
                        className={cn(
                          "ml-auto h-2 w-2 shrink-0 rounded-full",
                          readiness === "ready"
                            ? "bg-[var(--success)]"
                            : "bg-[var(--warning)]"
                        )}
                      />
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </aside>

        <section className="min-w-0">
          {tab === "agents" && agent && (
            <AgentsPanel
              doc={doc}
              agent={agent}
              agents={agents}
              selectedAgentId={agentId}
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

          {tab === "knowledge" && (
            <KnowledgeBasePanel
              doc={doc}
              onConfigureVectorDB={() => setVectordbOpen(true)}
            />
          )}

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
          "flex max-h-[85vh] w-full flex-col rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg)] shadow-xl",
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
          <button
            type="button"
            onClick={onClose}
            className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-1.5 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
          >
            {t("common.cancel")}
          </button>
          <button
            type="button"
            onClick={save}
            disabled={!dirty}
            className="rounded-[var(--radius-sm)] bg-[var(--accent,var(--text))] px-3 py-1.5 text-sm font-medium text-[var(--bg)] disabled:opacity-50"
          >
            {t("common.save")}
          </button>
        </>
      }
    >
      <p className="mb-3 text-xs text-[var(--text-muted)]">
        The agent's full system prompt (Markdown). This is its persona — tools and
        skills are appended automatically. Leave blank to use the built-in default.
      </p>
      <textarea
        autoFocus
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={20}
        placeholder="Leave blank to use the built-in default persona"
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
  const [text, setText] = useState<string>(doc.default_instructions ?? "");

  const commit = () => {
    if (text !== (doc.default_instructions ?? "")) {
      void onSave({ ...doc, default_instructions: text });
    }
  };

  return (
    <>
      <Header
        title="Default Persona"
        body="Control Room — the Markdown a new agent starts from. New agents copy this into their own Instructions at creation; editing it here doesn't change existing agents. Leave blank to seed new agents from the built-in default."
      />
      <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onBlur={commit}
          rows={18}
          placeholder="Leave blank to use the built-in default persona"
          className="w-full resize-y rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 font-mono text-xs leading-6"
        />
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
    <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4">
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

  const toggle = (kbId: string) => {
    const next = new Set(selected);
    if (next.has(kbId)) next.delete(kbId);
    else next.add(kbId);
    void onSave(
      applyAgentPatch(doc, agent.id, { knowledge: { kb_ids: [...next] } })
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
    </>
  );

  if (bare) return inner;

  return (
    <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4">
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
    <EditorDialog title={`Tools — ${agent.name}`} onClose={onClose} wide>
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
    <EditorDialog title={`Knowledge — ${agent.name}`} onClose={onClose}>
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
    <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="mb-2 flex items-center gap-2 text-[var(--text-muted)]">
        {icon}
        <h3 className="text-sm font-semibold text-[var(--text)]">{title}</h3>
        {badge}
        <button
          type="button"
          aria-label={editLabel}
          onClick={onEdit}
          className="ml-auto inline-flex items-center gap-1 rounded-[var(--radius-sm)] px-2 py-1 text-xs text-[var(--accent,var(--text))] hover:bg-[var(--surface-2)]"
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

/** Basic info — the agent's Name and Model, edited inline (no dialog). It's light
 * enough to sit on the page: Name is local and commits on blur; Model commits on
 * change. Parent keys this on agent.id so local state resets on switch. */
function BasicInfoSection({
  doc,
  agent,
  defaultModel,
  modelOptions,
  inherits,
  isDefault,
  onSave,
}: {
  doc: AgentConfigDocument;
  agent: AgentProfile;
  defaultModel: string;
  modelOptions: string[];
  inherits: boolean;
  isDefault: boolean;
  onSave: (doc: AgentConfigDocument, message?: string) => Promise<void>;
}) {
  const { t } = useI18n();
  const [name, setName] = useState(agent.name);
  const commitName = () => {
    if (name !== agent.name) void onSave(applyAgentPatch(doc, agent.id, { name }));
  };

  return (
    <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="mb-3 flex items-center gap-2 text-[var(--text-muted)]">
        <Settings2 className="h-4 w-4" />
        <h3 className="text-sm font-semibold text-[var(--text)]">{t("settings.basicInfo")}</h3>
        {isDefault && (
          <span className="rounded-full bg-[var(--surface-2)] px-2 py-0.5 text-[10px] font-medium text-[var(--text-muted)]">
            {t("settings.defaultAgent")}
          </span>
        )}
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="block text-sm">
          <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Name</span>
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            onBlur={commitName}
            className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm font-semibold"
          />
        </label>
        <label className="block text-sm">
          <span className="mb-1 flex items-center gap-2 text-xs font-medium text-[var(--text-muted)]">
            Model
            <span
              className={cn(
                "rounded-full px-2 py-0.5 text-[10px] font-medium",
                inherits
                  ? "bg-[var(--surface-2)] text-[var(--text-muted)]"
                  : "bg-[var(--accent-soft,var(--surface-2))] text-[var(--accent,var(--text))]"
              )}
            >
              {inherits ? "Using system default" : "Override"}
            </span>
          </span>
          <select
            aria-label="Model"
            value={agent.model}
            onChange={(event) =>
              void onSave(applyAgentPatch(doc, agent.id, { model: event.target.value }))
            }
            className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 font-mono text-xs"
          >
            <option value="">
              {defaultModel ? `Inherit (uses ${defaultModel})` : "Inherit (deployment default)"}
            </option>
            {modelOptions.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}

function AgentsPanel({
  doc,
  agent,
  agents,
  selectedAgentId,
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
  selectedAgentId: string;
  setSelectedAgentId: (id: string) => void;
  tools: ReturnType<typeof systemTools>;
  skills: CapabilityConfig[];
  loading: boolean;
  onSave: (doc: AgentConfigDocument, message?: string) => Promise<void>;
  onToggleTool: (toolId: string) => void;
  onToggleSkill: (skillId: string) => void;
}) {
  const { t } = useI18n();
  const enabledTools = new Set(agent.tools.include);
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
  const inherits = !agent.model;

  // Which focused editor dialog is open, and the two-step delete confirm. Both
  // reset when the selected agent changes.
  const [dialog, setDialog] = useState<
    null | "persona" | "tools" | "skills" | "knowledge"
  >(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  useEffect(() => {
    setDialog(null);
    setConfirmDelete(false);
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

  return (
    <>
      {/* Top toolbar: the agent switcher lives here alone, so it's the only thing
          occupying the top; the full-width, read-only config sits below it. */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <UserRound className="h-4 w-4 text-[var(--text-muted)]" />
        <select
          aria-label="Select agent"
          value={selectedAgentId}
          onChange={(event) => setSelectedAgentId(event.target.value)}
          className="min-w-[200px] rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm font-medium"
        >
          {agents.map((item) => (
            <option key={item.id} value={item.id}>
              {item.name}
              {doc.default_agent === item.id ? t("settings.defaultSuffix") : ""}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => {
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
          }}
          className="inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-2 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
        >
          <Plus className="h-4 w-4" />
          New agent
        </button>
        <div className="flex-1" />
        {!isDefault && (
          <button
            type="button"
            onClick={makeDefault}
            className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-2 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
          >
            {t("settings.setDefault")}
          </button>
        )}
        {confirmDelete ? (
          <>
            <button
              type="button"
              onClick={deleteAgent}
              className="rounded-[var(--radius-sm)] border border-[var(--danger,#dc2626)] px-3 py-2 text-sm font-medium text-[var(--danger,#dc2626)] hover:bg-[var(--surface-2)]"
            >
              {t("settings.confirmDelete")}
            </button>
            <button
              type="button"
              onClick={() => setConfirmDelete(false)}
              className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-2 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
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
            className="inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-2 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)] disabled:opacity-50"
          >
            <Trash2 className="h-4 w-4" />
            {t("common.delete")}
          </button>
        )}
      </div>

      <div className="space-y-3">
        {/* Basic info — name + model, edited inline (light enough for no dialog). */}
        <BasicInfoSection
          key={`basic-${agent.id}`}
          doc={doc}
          agent={agent}
          defaultModel={defaultModel}
          modelOptions={modelOptions}
          inherits={inherits}
          isDefault={isDefault}
          onSave={onSave}
        />

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
        <div className="grid gap-3 sm:grid-cols-3">
          <PreviewCard
            icon={<Wrench className="h-4 w-4" />}
            title={t("settings.tools")}
            badge={
              <span className="rounded-full bg-[var(--surface-2)] px-2 py-0.5 text-[10px] font-medium text-[var(--text-muted)]">
                {enabledTools.size}/{tools.length}
              </span>
            }
            editLabel={t("settings.editTools")}
            onEdit={() => setDialog("tools")}
          >
            <p className="line-clamp-2 text-xs text-[var(--text-muted)]">
              {previewNames(toolNames, t("settings.noToolsEnabled"))}
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
            title={t("settings.knowledge")}
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
  return (
    <>
      <Header
        title="System Tools"
        body="Control Room — configure which tools exist in the deployment. Each agent decides in Agent Studio whether to use them."
      />
      <div className="grid gap-3 lg:grid-cols-3">
        {tools.map((cap) => (
          <div
            key={cap.id}
            className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4"
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
                className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-1.5 text-xs hover:bg-[var(--surface-2)] disabled:opacity-60"
              >
                {cap.enabled ? "Disable" : "Enable"}
              </button>
              {cap.id === "search" && (
                <button
                  type="button"
                  onClick={onConfigureSearch}
                  className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-1.5 text-xs hover:bg-[var(--surface-2)]"
                >
                  Configure
                </button>
              )}
              {cap.id === "sandbox" && (
                <button
                  type="button"
                  onClick={onConfigureSandbox}
                  className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-1.5 text-xs hover:bg-[var(--surface-2)]"
                >
                  Configure
                </button>
              )}
              {cap.id === "knowledge" && (
                <button
                  type="button"
                  onClick={onConfigureVectorDB}
                  className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-1.5 text-xs hover:bg-[var(--surface-2)]"
                >
                  Vector DB
                </button>
              )}
              {cap.id === "aliyun_pai" && (
                <button
                  type="button"
                  onClick={onConfigureAliyun}
                  className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-1.5 text-xs hover:bg-[var(--surface-2)]"
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
  return (
    <>
      <div className="mb-5 flex items-start justify-between gap-3">
        <Header
          title="System Skills"
          body="Control Room — install and manage the deployment's skill library. Agents opt into skills in Agent Studio once dependencies are available."
        />
        <button
          type="button"
          onClick={onInstall}
          className="mt-1 flex shrink-0 items-center gap-2 rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-1.5 text-xs font-medium text-white hover:opacity-90"
        >
          <Upload className="h-3.5 w-3.5" />
          Install
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
                {skill.dependencies.length ? `Requires: ${skill.dependencies.join(", ")}` : "No dependency"}
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
              {skill.enabled ? "Installed" : "Disabled"}
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
      <div className="w-full max-w-lg rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] shadow-xl">
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
                className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm"
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
                className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm"
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
            className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-1.5 text-xs text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={loading || submitting}
            onClick={() => void submit()}
            className="rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-1.5 text-xs font-medium text-white disabled:opacity-60"
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
        className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm"
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
  return (
    <>
      <Header
        title="YAML"
        body="Control Room — advanced raw configuration for system capabilities and agent profiles. Secrets are masked when read from the API."
      />
      <div className="mb-3 flex justify-end gap-2">
        <button
          type="button"
          onClick={() => void onLoad()}
          className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-1.5 text-xs text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
        >
          Reload
        </button>
        <button
          type="button"
          disabled={loading}
          onClick={() => void onSave()}
          className="rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-1.5 text-xs font-medium text-white disabled:opacity-60"
        >
          Save YAML
        </button>
      </div>
      <textarea
        value={yamlText}
        onChange={(event) => setYamlText(event.target.value)}
        spellCheck={false}
        className="min-h-[560px] w-full resize-y rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4 font-mono text-xs leading-5 text-[var(--text)] outline-none"
      />
    </>
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
      <div className="w-full max-w-xl rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg)] shadow-xl">
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
              className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
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
              className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
            />
          </label>
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">API Key Env</span>
            <input
              value={apiKeyEnv}
              placeholder={providerName === "tavily" ? "TAVILY_API_KEY" : "BRAVE_SEARCH_API_KEY"}
              onChange={(event) => setApiKeyEnv(event.target.value)}
              className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
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
              className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
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
              className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
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
            className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-1.5 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)] disabled:opacity-50"
          >
            {testing ? "Testing..." : "Test search"}
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-[var(--radius-sm)] px-3 py-1.5 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={loading}
            onClick={saveSearch}
            className="rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-1.5 text-sm font-medium text-white disabled:opacity-60"
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
      <div className="w-full max-w-xl rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg)] shadow-xl">
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
                  className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
                />
              </label>
              <label className="block text-sm">
                <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Index Prefix</span>
                <input
                  value={indexPrefix}
                  placeholder="kb"
                  onChange={(event) => setIndexPrefix(event.target.value)}
                  className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
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
                  className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
                />
              </label>
              <label className="block text-sm">
                <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">API Key Env</span>
                <input
                  value={apiKeyEnv}
                  placeholder="ELASTICSEARCH_API_KEY"
                  onChange={(event) => setApiKeyEnv(event.target.value)}
                  className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
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
                  className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
                />
              </label>
              <label className="block text-sm">
                <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Password</span>
                <input
                  value={password}
                  type="password"
                  placeholder={vdb.password ? "Already configured" : "Paste password or use env below"}
                  onChange={(event) => setPassword(event.target.value)}
                  className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
                />
              </label>
              <label className="block text-sm">
                <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Password Env</span>
                <input
                  value={passwordEnv}
                  placeholder="ELASTICSEARCH_PASSWORD"
                  onChange={(event) => setPasswordEnv(event.target.value)}
                  className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
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
                  className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
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
            className="rounded-[var(--radius-sm)] px-3 py-1.5 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={loading}
            onClick={saveVectorDB}
            className="rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-1.5 text-sm font-medium text-white disabled:opacity-60"
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
          include: Array.from(new Set([...agent.tools.include, "code_sandbox"])),
          exclude: agent.tools.exclude.filter((name) => name !== "code_sandbox"),
        },
      })),
    };
    await onSave(next);
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/30 p-4">
      <div className="max-h-[92vh] w-full max-w-3xl overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg)] shadow-xl">
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
                className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
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
                className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
              />
            </Field>
            <Field label="Gateway API key">
              <input
                aria-label="Sandbox gateway API key"
                value={apiKey}
                type="password"
                placeholder={provider?.secret_configured ? "Already configured" : "Optional if env is set"}
                onChange={(event) => setApiKey(event.target.value)}
                className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
              />
            </Field>
            <Field label="Gateway API key env">
              <input
                aria-label="Sandbox gateway API key env"
                value={apiKeyEnv}
                placeholder="AGENTRUN_SANDBOX_API_KEY"
                onChange={(event) => setApiKeyEnv(event.target.value)}
                className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
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
                className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
              />
            </Field>
            <Field label="Account ID env">
              <input
                aria-label="Sandbox account ID env"
                value={accountIdEnv}
                placeholder="AGENTRUN_ACCOUNT_ID"
                onChange={(event) => setAccountIdEnv(event.target.value)}
                className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
              />
            </Field>
            <Field label="Session idle seconds">
              <input
                aria-label="Sandbox session idle seconds"
                value={sessionIdle}
                type="number"
                min={30}
                onChange={(event) => setSessionIdle(event.target.value)}
                className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
              />
            </Field>
            <Field label="Execution timeout seconds">
              <input
                aria-label="Sandbox execution timeout seconds"
                value={timeout}
                type="number"
                min={1}
                onChange={(event) => setTimeoutValue(event.target.value)}
                className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
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
            className="rounded-[var(--radius-sm)] px-3 py-1.5 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={loading}
            onClick={saveSandbox}
            className="rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-1.5 text-sm font-medium text-white disabled:opacity-60"
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
