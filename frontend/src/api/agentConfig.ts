import { apiFetch } from "../lib/apiFetch";

export type Permission = "disabled" | "ask" | "auto" | "admin";
export type CapabilityStatus = "ready" | "missing_config" | "error" | "disabled";
export type ProviderStatus = "untested" | "healthy" | "missing_config" | "error";
export type SetupMode = "local_first" | "cloud_enhanced" | "developer";

export interface ProviderConfig {
  id: string;
  type: "llm" | "search" | "embedding" | "rerank" | "vectordb" | "sandbox" | "cloud_auth";
  name: string;
  status: ProviderStatus;
  settings: Record<string, unknown>;
  secret_configured: boolean;
  error?: string | null;
  used_by: string[];
}

export interface CapabilityConfig {
  id: string;
  kind: "core_tool" | "skill";
  name: string;
  description: string;
  enabled: boolean;
  permission: Permission;
  status: CapabilityStatus;
  dependencies: string[];
  provider_refs: string[];
  settings: Record<string, unknown>;
  error?: string | null;
}

export interface SetupConfig {
  completed: boolean;
  completed_at?: string | null;
  mode?: SetupMode | null;
  skipped_steps: string[];
}

// Global vector-store selection (singleton — not per-KB). Secrets mask to
// "********" on read; leave a masked field untouched to preserve the stored value.
export interface VectorDBConfig {
  engine: "local" | "elasticsearch";
  url: string;
  index_prefix: string;
  api_key: string;
  api_key_env: string;
  username: string;
  password: string;
  password_env: string;
  verify_certs: boolean;
  timeout: number;
  status: ProviderStatus;
  secret_configured: boolean;
  error?: string | null;
}

export interface KnowledgeBaseConfig {
  vectordb: VectorDBConfig;
}

// The LLM/embedding/rerank model catalog (persisted as config.yaml `models:` and
// parsed by the backend `ModelCatalog`). A provider holds the shared connection
// (base_url + the NAME of the env var that carries its API key — never an inline
// secret, which the models section does not mask); a model is a registration under
// a provider carrying a `type`. Index signatures preserve fields we don't model so
// they round-trip through PUT /v1/config untouched.
export interface ModelSpecDoc {
  id: string;
  type?: "chat" | "embedding" | "rerank";
  protocol?: "openai" | "dashscope";
  dimension?: number;
  base_url?: string;
  [k: string]: unknown;
}

export interface ModelProviderDoc {
  name: string;
  base_url?: string;
  api_key_env?: string;
  models?: ModelSpecDoc[];
  [k: string]: unknown;
}

export interface ModelCatalogDoc {
  default_model?: string;
  default_embedding_model?: string;
  default_rerank_model?: string;
  providers?: ModelProviderDoc[];
  [k: string]: unknown;
}

export interface AgentToolsConfig {
  include: string[];
  exclude: string[];
}

export interface AgentSkillsConfig {
  enabled: string[];
}

export interface SkillLibraryConfig {
  root?: string;
  mount?: Record<string, unknown>;
  install?: Record<string, unknown>;
  dependencies?: Record<string, unknown>;
  config?: Record<string, unknown>;
  installed?: Array<Record<string, unknown>>;
}

// Per-agent knowledge scoping. `kb_ids` is a soft default: when non-empty the
// agent's knowledge tools default to these bases (still permission-checked per
// request server-side). Empty = search every base the user can access.
export interface AgentKnowledgeConfig {
  kb_ids: string[];
}

export interface AgentProfile {
  id: string;
  name: string;
  description: string;
  model: string;
  // The agent's full system prompt (freeform Markdown). This IS its persona/base
  // prompt; tools and skills are appended automatically. Blank => built-in default.
  instructions: string;
  // Per-agent knowledge scoping (soft default; still permission-checked per KB).
  knowledge: AgentKnowledgeConfig;
  // Markdown describing the repos under the read-only /mnt/code layer; injected
  // into the system prompt when that layer is mounted. Can be AI-generated.
  code_manifest: string;
  tools: AgentToolsConfig;
  skills: AgentSkillsConfig;
  settings: Record<string, unknown>;
}

export interface AgentConfigDocument {
  setup: SetupConfig;
  models: ModelCatalogDoc;
  knowledgebase: KnowledgeBaseConfig;
  skills: SkillLibraryConfig;
  // The admin-editable "Default Persona" template. Its Markdown seeds a new
  // agent's `instructions` at creation (a snapshot copy — editing it never
  // touches existing agents, and it is never merged into them at runtime).
  default_instructions: string;
  default_agent: string;
  agents: AgentProfile[];
  providers: ProviderConfig[];
  capabilities: CapabilityConfig[];
}

/** A fresh, blank agent profile seeded from the deployment's "Default Persona"
 * template (`doc.default_instructions`). The snapshot is taken here at creation —
 * a later edit to the template never reaches this agent. Blank template => the
 * agent's own `instructions` stays blank and falls back to the built-in default. */
export function newAgentProfile(
  doc: AgentConfigDocument,
  id: string,
  name: string
): AgentProfile {
  return {
    id,
    name,
    description: "",
    model: "",
    instructions: doc.default_instructions || "",
    knowledge: { kb_ids: [] },
    code_manifest: "",
    tools: { include: [], exclude: [] },
    skills: { enabled: [] },
    settings: {},
  };
}

export interface SkillUploadResult {
  upload_id: string;
  filename?: string;
  size: number;
}

export interface SkillInstallSource {
  type: "zip_upload" | "url" | "git";
  upload_id?: string;
  url?: string;
  checksum?: string;
  ref?: string;
  path?: string;
}

export interface SkillInstallResult {
  ok: boolean;
  result: Record<string, unknown>;
  config: AgentConfigDocument;
}

export interface SkillEnableResult {
  ok: boolean;
  result: Record<string, unknown>;
  config: AgentConfigDocument;
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `request failed: ${res.status}`;
    try {
      const body = await res.json();
      if (body?.error?.message) {
        message = body.error.message;
      } else if (typeof body?.detail === "string") {
        // FastAPI HTTPException(detail=...) shape.
        message = body.detail;
      } else if (Array.isArray(body?.detail)) {
        // Pydantic validation error list.
        message = body.detail
          .map((e: { loc?: unknown[]; msg?: string }) => {
            const loc = Array.isArray(e?.loc) ? e.loc.join(".") : "?";
            return `${loc}: ${e?.msg ?? ""}`;
          })
          .join("; ");
      }
    } catch {
      /* response body not JSON */
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export async function getSetup(): Promise<AgentConfigDocument> {
  return jsonOrThrow(await apiFetch("/v1/setup"));
}

export async function saveSetup(setup: SetupConfig): Promise<AgentConfigDocument> {
  return jsonOrThrow(
    await apiFetch("/v1/setup", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(setup),
    })
  );
}

export async function getAgentConfig(): Promise<AgentConfigDocument> {
  return jsonOrThrow(await apiFetch("/v1/config"));
}

export async function saveAgentConfig(
  doc: AgentConfigDocument
): Promise<AgentConfigDocument> {
  return jsonOrThrow(
    await apiFetch("/v1/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(doc),
    })
  );
}

/** Ask the backend to explore the sandbox /mnt/code layer with the LLM and
 * return a generated Markdown manifest. Not persisted — the caller reviews it
 * and saves it onto the agent profile via saveAgentConfig. */
export async function generateCodeManifest(
  agentId: string
): Promise<{ manifest: string }> {
  return jsonOrThrow(
    await apiFetch(`/v1/agents/${encodeURIComponent(agentId)}/code-manifest/generate`, {
      method: "POST",
    })
  );
}

export async function getAgentConfigYaml(): Promise<string> {
  const res = await apiFetch("/v1/config.yaml");
  if (!res.ok) throw new Error(`config YAML request failed: ${res.status}`);
  return res.text();
}

export async function saveAgentConfigYaml(yaml: string): Promise<AgentConfigDocument> {
  return jsonOrThrow(
    await apiFetch("/v1/config.yaml", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml }),
    })
  );
}

export async function testSearchProvider(
  query: string,
  numResults = 3
): Promise<{ ok: boolean; output: string }> {
  return jsonOrThrow(
    await apiFetch("/v1/config/search/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, num_results: numResults }),
    })
  );
}

// Probe an LLM connection: resolves its key env, opens the client, streams a
// one-token completion. Tests the *saved* catalog — save the connection first.
// `model` is a `provider/model-id` ref; omit to test the deployment default.
export async function testModelConnection(
  model?: string
): Promise<{ ok: boolean; output: string }> {
  return jsonOrThrow(
    await apiFetch("/v1/config/models/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    })
  );
}

export async function uploadSkillZip(file: File): Promise<SkillUploadResult> {
  const form = new FormData();
  form.append("file", file);
  return jsonOrThrow(
    await apiFetch("/v1/skills/uploads", {
      method: "POST",
      body: form,
    })
  );
}

export async function installSkill(payload: {
  source: SkillInstallSource;
  enable_for_agent?: string;
  enable_after_build?: boolean;
  overwrite?: boolean;
}): Promise<SkillInstallResult> {
  return jsonOrThrow(
    await apiFetch("/v1/skills/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

export async function enableSkillForAgent(payload: {
  skill_id: string;
  agent_id?: string;
  enabled?: boolean;
}): Promise<SkillEnableResult> {
  return jsonOrThrow(
    await apiFetch("/v1/skills/enable", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

// --------------------------------------------------------------------------- //
// Aliyun PAI cross-account authorization (per-user, keyed on user_id).
// --------------------------------------------------------------------------- //
export interface AliyunRegionResult {
  region: string;
  ok: boolean;
  pai_total?: number | null;
  error_code?: string | null;
}

export interface AliyunVerdict {
  ok: boolean;
  stage?: string;
  account_id?: string | null;
  caller_arn?: string | null;
  pai_total?: number | null;
  error_code?: string | null;
  error_message?: string | null;
  // Per-region discovery snapshot (STS creds are global; services are per-region).
  regions?: AliyunRegionResult[] | null;
}

export interface AliyunStatus {
  bound: boolean;
  region: string;
  // Regions where the binding is reachable / has services, from last authorize.
  regions?: string[] | null;
  service_regions?: string[] | null;
  region_totals?: Record<string, number> | null;
  default_region?: string | null;
  external_id: string | null;
  role_arn?: string | null;
  assumed_account_id?: string | null;
  verified_at?: string | null;
  ros_url?: string | null;
  configured: boolean;
}

export interface AliyunAuthorizeResult {
  ok: boolean;
  external_id: string;
  verdict: AliyunVerdict;
}

// Aliyun binding is keyed on the authenticated user server-side; the client
// passes only the role_arn.
export async function getAliyunStatus(): Promise<AliyunStatus> {
  return jsonOrThrow(await apiFetch("/v1/aliyun/status"));
}

export async function authorizeAliyun(payload: {
  role_arn: string;
}): Promise<AliyunAuthorizeResult> {
  return jsonOrThrow(
    await apiFetch("/v1/aliyun/authorize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

export async function deauthorizeAliyun(): Promise<{ ok: boolean; bound: boolean }> {
  return jsonOrThrow(
    await apiFetch("/v1/aliyun/deauthorize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    })
  );
}

// Re-run AssumeRole + PAI probing against the EXISTING binding (no re-authorize).
// Health-check for "authorized but the CLI still errors"; 404 if nothing bound.
export async function verifyAliyun(): Promise<AliyunAuthorizeResult> {
  return jsonOrThrow(
    await apiFetch("/v1/aliyun/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    })
  );
}
