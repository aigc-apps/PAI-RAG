export type Permission = "disabled" | "ask" | "auto" | "admin";
export type CapabilityStatus = "ready" | "missing_config" | "error" | "disabled";
export type ProviderStatus = "untested" | "healthy" | "missing_config" | "error";
export type SetupMode = "local_first" | "cloud_enhanced" | "developer";

export interface ProviderConfig {
  id: string;
  type: "llm" | "search" | "embedding" | "rerank" | "vectordb" | "sandbox";
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

export interface AgentProfile {
  id: string;
  name: string;
  description: string;
  model: string;
  instructions: string;
  tools: AgentToolsConfig;
  skills: AgentSkillsConfig;
  settings: Record<string, unknown>;
}

export interface AgentConfigDocument {
  setup: SetupConfig;
  models: Record<string, unknown>;
  skills: SkillLibraryConfig;
  default_agent: string;
  agents: AgentProfile[];
  providers: ProviderConfig[];
  capabilities: CapabilityConfig[];
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
  return jsonOrThrow(await fetch("/v1/setup"));
}

export async function saveSetup(setup: SetupConfig): Promise<AgentConfigDocument> {
  return jsonOrThrow(
    await fetch("/v1/setup", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(setup),
    })
  );
}

export async function getAgentConfig(): Promise<AgentConfigDocument> {
  return jsonOrThrow(await fetch("/v1/config"));
}

export async function saveAgentConfig(
  doc: AgentConfigDocument
): Promise<AgentConfigDocument> {
  return jsonOrThrow(
    await fetch("/v1/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(doc),
    })
  );
}

export async function getAgentConfigYaml(): Promise<string> {
  const res = await fetch("/v1/config.yaml");
  if (!res.ok) throw new Error(`config YAML request failed: ${res.status}`);
  return res.text();
}

export async function saveAgentConfigYaml(yaml: string): Promise<AgentConfigDocument> {
  return jsonOrThrow(
    await fetch("/v1/config.yaml", {
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
    await fetch("/v1/config/search/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, num_results: numResults }),
    })
  );
}

export async function uploadSkillZip(file: File): Promise<SkillUploadResult> {
  const form = new FormData();
  form.append("file", file);
  return jsonOrThrow(
    await fetch("/v1/skills/uploads", {
      method: "POST",
      headers: { "X-Admin": "true" },
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
    await fetch("/v1/skills/install", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Admin": "true" },
      body: JSON.stringify(payload),
    })
  );
}
