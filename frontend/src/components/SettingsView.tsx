import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  ArrowLeft,
  CircleAlert,
  Database,
  GitBranch,
  Globe2,
  ShieldCheck,
  Settings2,
  Terminal,
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
import { cn } from "../lib/cn";
import { useAgentConfigStore } from "../store/agentConfig";
import { useAliyunDialog } from "../store/aliyunDialog";
import { ModelsPanel } from "./ModelsPanel";
import { KnowledgeBasePanel } from "./KnowledgeBasePanel";
import { ThemeToggle } from "./ThemeToggle";

type Tab = "agents" | "tools" | "models" | "knowledge" | "skills" | "providers" | "yaml";

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

  const tabs: Array<{ id: Tab; label: string }> = [
    { id: "agents", label: "Agents" },
    { id: "tools", label: "Tools" },
    { id: "models", label: "Models" },
    { id: "knowledge", label: "Knowledge Base" },
    { id: "skills", label: "Skills" },
    { id: "providers", label: "Providers" },
    { id: "yaml", label: "YAML" },
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
        <aside className="space-y-1">
          {tabs.map((item) => (
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
            </button>
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

          {tab === "models" && <ModelsPanel doc={doc} />}

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

          {tab === "providers" && (
            <ProvidersPanel doc={doc} onEditYaml={openYaml} />
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
  const enabledTools = new Set(agent.tools.include);
  const enabledSkills = new Set(agent.skills.enabled);
  return (
    <>
      <Header
        title="Agents"
        body="Configure how each agent uses system capabilities. Tools and providers are configured globally; agents choose which of them to use."
      />
      <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-2">
          {agents.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setSelectedAgentId(item.id)}
              className={cn(
                "flex w-full items-center gap-2 rounded-[var(--radius-sm)] px-3 py-2 text-left text-sm",
                selectedAgentId === item.id
                  ? "bg-[var(--surface-2)] text-[var(--text)]"
                  : "text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
              )}
            >
              <UserRound className="h-4 w-4" />
              <span>{item.name}</span>
              {doc.default_agent === item.id && (
                <span className="ml-auto text-xs text-[var(--text-faint)]">default</span>
              )}
            </button>
          ))}
        </div>

        <div className="space-y-4">
          <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4">
            <div className="mb-4 flex items-center gap-2">
              <Settings2 className="h-4 w-4 text-[var(--text-muted)]" />
              <h3 className="text-sm font-semibold">Profile</h3>
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              <label className="block text-sm">
                <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Name</span>
                <input
                  value={agent.name}
                  onChange={(event) =>
                    void onSave(applyAgentPatch(doc, agent.id, { name: event.target.value }))
                  }
                  className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2"
                />
              </label>
              <label className="block text-sm">
                <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Model</span>
                <input
                  value={agent.model}
                  onChange={(event) =>
                    void onSave(applyAgentPatch(doc, agent.id, { model: event.target.value }))
                  }
                  className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 font-mono text-xs"
                />
              </label>
            </div>
            <label className="mt-3 block text-sm">
              <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">Instructions</span>
              <textarea
                value={agent.instructions}
                onChange={(event) =>
                  void onSave(applyAgentPatch(doc, agent.id, { instructions: event.target.value }))
                }
                rows={4}
                className="w-full resize-none rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm"
              />
            </label>
          </div>

          <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4">
            <h3 className="mb-3 text-sm font-semibold">Tools For This Agent</h3>
            <div className="grid gap-2 md:grid-cols-2">
              {tools.map((tool) => (
                <button
                  key={tool.id}
                  type="button"
                  disabled={loading || !tool.available}
                  onClick={() => onToggleTool(tool.id)}
                  className={cn(
                    "rounded-[var(--radius-sm)] border px-3 py-2 text-left text-sm",
                    enabledTools.has(tool.id)
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
          </div>

          <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4">
            <h3 className="mb-3 text-sm font-semibold">Skills For This Agent</h3>
            <div className="grid gap-2 md:grid-cols-2">
              {skills.map((skill) => (
                <button
                  key={skill.id}
                  type="button"
                  disabled={loading || skill.status !== "ready"}
                  onClick={() => onToggleSkill(skill.id)}
                  className={cn(
                    "rounded-[var(--radius-sm)] border px-3 py-2 text-left text-sm",
                    enabledSkills.has(skill.id)
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
            {skillSummary(doc, agent).length > 0 && (
              <div className="mt-3 text-xs text-[var(--text-faint)]">
                Enabled: {skillSummary(doc, agent).map((skill) => skill.name).join(", ")}
              </div>
            )}
          </div>
        </div>
      </div>
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
        body="Configure which tools exist in the deployment. Agent profiles decide whether each agent can use them."
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
          body="Skills are reusable higher-level behaviors. Agents opt into skills after their dependencies are available."
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

function ProvidersPanel({
  doc,
  onEditYaml,
}: {
  doc: AgentConfigDocument;
  onEditYaml: () => void;
}) {
  return (
    <>
      <Header
        title="Providers"
        body="Providers hold deployment-wide credentials and endpoints. Tools and agents reference these services indirectly."
      />
      <div className="mb-3 flex justify-end">
        <button
          type="button"
          onClick={onEditYaml}
          className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-1.5 text-xs text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
        >
          Edit YAML
        </button>
      </div>
      <div className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)]">
        {doc.providers.map((provider) => (
          <div
            key={provider.id}
            className="grid gap-2 border-b border-[var(--border)] px-4 py-3 text-sm last:border-b-0 md:grid-cols-[1fr_140px_1fr]"
          >
            <div>
              <div className="font-medium">{provider.name}</div>
              <div className="text-xs text-[var(--text-faint)]">{provider.type}</div>
            </div>
            <div className={cn("text-xs md:text-sm", statusClass(provider.status))}>
              {provider.status}
            </div>
            <div className="text-xs text-[var(--text-muted)]">
              Used by: {provider.used_by.join(", ") || "none"}
            </div>
          </div>
        ))}
      </div>
    </>
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
        body="Advanced configuration for system capabilities and agent profiles. Secrets are masked when read from the API."
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
            知识库检索使用 Elasticsearch。填写连接信息并保存后立即全局生效。
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
                认证：填写 API Key，或用户名 + 密码（二选一，优先使用 API Key）。密钥可直接填写或用环境变量名引用。
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
            向量库为全局设置。修改连接后，已建知识库需重新索引才能在新存储中检索。
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
