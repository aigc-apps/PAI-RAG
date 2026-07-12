import { useState } from "react";
import { Plus, Trash2, Zap, Loader2, CircleCheck, CircleAlert } from "lucide-react";
import { toast } from "sonner";
import type {
  AgentConfigDocument,
  ModelCatalogDoc,
  ModelProviderDoc,
  ModelSpecDoc,
} from "../api/agentConfig";
import { testModelConnection } from "../api/agentConfig";
import { cn } from "../lib/cn";
import { useAgentConfigStore } from "../store/agentConfig";
import { useI18n, type MessageKey } from "../i18n";

// A model's type as the backend `ModelSpec` stores it; the UI labels "chat" as
// "LLM" but the wire value stays `chat`.
type ModelType = "chat" | "embedding" | "rerank";

const TYPE_LABEL_KEY: Record<ModelType, MessageKey> = {
  chat: "conn.typeLLM",
  embedding: "conn.typeEmbedding",
  rerank: "conn.typeRerank",
};

/** Reference format used everywhere the catalog points at a model. */
const ref = (provider: string, id: string) => `${provider}/${id}`;

const providersOf = (cat: ModelCatalogDoc): ModelProviderDoc[] =>
  cat.providers ?? [];

/** The first chat model under a provider — what its Test button probes. */
function firstChatRef(p: ModelProviderDoc): string | null {
  const chat = (p.models ?? []).find((m) => (m.type ?? "chat") === "chat");
  return chat ? ref(p.name, chat.id) : null;
}

/** Which `default_*` field a model of the given type sets. */
function defaultKeyFor(type: ModelType): keyof ModelCatalogDoc {
  if (type === "embedding") return "default_embedding_model";
  if (type === "rerank") return "default_rerank_model";
  return "default_model";
}

/** Clear any `default_*` that points at a model no longer in the catalog. */
function clearDefaultsPointingAt(
  cat: ModelCatalogDoc,
  predicate: (r: string) => boolean
): ModelCatalogDoc {
  const out = { ...cat };
  for (const key of [
    "default_model",
    "default_embedding_model",
    "default_rerank_model",
  ] as const) {
    const value = out[key];
    if (typeof value === "string" && predicate(value)) out[key] = undefined;
  }
  return out;
}

export function ConnectionsPanel({ doc }: { doc: AgentConfigDocument }) {
  const { t } = useI18n();
  const save = useAgentConfigStore((s) => s.save);
  const loading = useAgentConfigStore((s) => s.loading);
  const cat = doc.models ?? {};
  const providers = providersOf(cat);

  const saveCatalog = async (next: ModelCatalogDoc, onOk?: () => void) => {
    try {
      await save({ ...doc, models: next });
      onOk?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("conn.saveFailed"));
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold tracking-tight">{t("conn.title")}</h2>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--text-muted)]">
          {t("conn.intro.a")}
          <strong>{t("conn.intro.connection")}</strong>
          {t("conn.intro.b")}
          <strong>{t("conn.intro.models")}</strong>
          {t("conn.intro.c")}
          <code className="font-mono">provider/model-id</code>
          {t("conn.intro.d")}
          <strong>{t("conn.intro.test")}</strong>
          {t("conn.intro.e")}
          <strong>{t("conn.intro.kb")}</strong>
          {t("conn.intro.f")}
        </p>
      </div>

      <ConnectionsSection
        providers={providers}
        loading={loading}
        onSave={saveCatalog}
        cat={cat}
      />

      <ModelsSection
        providers={providers}
        loading={loading}
        onSave={saveCatalog}
        cat={cat}
      />
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Part 1 — Connections (model endpoints + credentials + Test)
// --------------------------------------------------------------------------- //
function ConnectionsSection({
  providers,
  loading,
  cat,
  onSave,
}: {
  providers: ModelProviderDoc[];
  loading: boolean;
  cat: ModelCatalogDoc;
  onSave: (next: ModelCatalogDoc, onOk?: () => void) => Promise<void>;
}) {
  const { t } = useI18n();
  const [name, setName] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKeyEnv, setApiKeyEnv] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  // Per-connection Test result keyed by provider name.
  const [testing, setTesting] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, { ok: boolean; output: string }>>({});

  const reset = () => {
    setName("");
    setBaseUrl("");
    setApiKeyEnv("");
    setEditing(null);
  };

  const edit = (p: ModelProviderDoc) => {
    setEditing(p.name);
    setName(p.name);
    setBaseUrl(p.base_url ?? "");
    setApiKeyEnv(p.api_key_env ?? "");
  };

  const submit = () => {
    const trimmed = name.trim();
    if (!trimmed) {
      toast.error(t("conn.nameRequired"));
      return;
    }
    // Upsert by name. When editing, the name field is locked so `editing`
    // always matches an existing row; a new name always appends.
    const list = [...(cat.providers ?? [])];
    const idx = list.findIndex((p) => p.name === (editing ?? trimmed));
    const patch = {
      base_url: baseUrl.trim() || undefined,
      api_key_env: apiKeyEnv.trim() || undefined,
    };
    if (idx >= 0) {
      list[idx] = { ...list[idx], ...patch };
    } else {
      list.push({ name: trimmed, ...patch, models: [] });
    }
    void onSave({ ...cat, providers: list }, reset);
  };

  const remove = (p: ModelProviderDoc) => {
    // Deleting a connection cascades its models — clear any default that pointed
    // at one of them so the saved catalog stays valid.
    const providers = (cat.providers ?? []).filter((x) => x.name !== p.name);
    const next = clearDefaultsPointingAt(
      { ...cat, providers },
      (r) => r.startsWith(`${p.name}/`)
    );
    void onSave(next);
    if (editing === p.name) reset();
  };

  const test = async (p: ModelProviderDoc) => {
    const modelRef = firstChatRef(p);
    if (!modelRef) {
      setResult((r) => ({
        ...r,
        [p.name]: { ok: false, output: t("conn.registerLlmFirst") },
      }));
      return;
    }
    setTesting(p.name);
    try {
      const res = await testModelConnection(modelRef);
      setResult((r) => ({ ...r, [p.name]: res }));
    } catch (err) {
      setResult((r) => ({
        ...r,
        [p.name]: { ok: false, output: err instanceof Error ? err.message : t("conn.testFailed") },
      }));
    } finally {
      setTesting(null);
    }
  };

  return (
    <section className="space-y-3">
      <div>
        <h3 className="text-sm font-semibold">{t("conn.section1Title")}</h3>
        <p className="mt-1 text-xs text-[var(--text-muted)]">
          {t("conn.section1Hint.a")}
          <code className="font-mono">provider/model-id</code>
          {t("conn.section1Hint.b")}
          <em>{t("conn.wordName")}</em>
          {t("conn.section1Hint.c")}
        </p>
      </div>

      <div className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)]">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--border)] text-left text-xs text-[var(--text-muted)]">
              <th className="px-3 py-2 font-medium">{t("conn.colName")}</th>
              <th className="px-3 py-2 font-medium">{t("conn.colBaseUrl")}</th>
              <th className="px-3 py-2 font-medium">{t("conn.colApiKeyEnv")}</th>
              <th className="px-3 py-2 font-medium">{t("conn.colModels")}</th>
              <th className="px-3 py-2" />
            </tr>
          </thead>
          <tbody>
            {providers.map((p) => {
              const res = result[p.name];
              return (
                <tr key={p.name} className="border-b border-[var(--border)] last:border-0">
                  <td className="px-3 py-2 font-medium">{p.name}</td>
                  <td className="px-3 py-2 font-mono text-xs text-[var(--text-muted)]">
                    {p.base_url || "—"}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs text-[var(--text-muted)]">
                    {p.api_key_env || "—"}
                  </td>
                  <td className="px-3 py-2 text-xs text-[var(--text-muted)]">
                    {(p.models ?? []).length}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <div className="flex flex-col items-end gap-1">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          type="button"
                          aria-label={t("conn.testConnAria", { name: p.name })}
                          disabled={testing === p.name}
                          onClick={() => void test(p)}
                          className="inline-flex items-center gap-1 rounded-[var(--radius-sm)] border border-[var(--border)] px-2 py-1 text-xs hover:bg-[var(--surface-2)] disabled:opacity-60"
                        >
                          {testing === p.name ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          ) : (
                            <Zap className="h-3.5 w-3.5" />
                          )}
                          {t("conn.test")}
                        </button>
                        <button
                          type="button"
                          onClick={() => edit(p)}
                          className="rounded-[var(--radius-sm)] border border-[var(--border)] px-2 py-1 text-xs hover:bg-[var(--surface-2)]"
                        >
                          {t("common.edit")}
                        </button>
                        <button
                          type="button"
                          aria-label={t("conn.deleteConnAria", { name: p.name })}
                          onClick={() => remove(p)}
                          className="rounded-[var(--radius-sm)] border border-[var(--border)] p-1.5 text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--danger)]"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                      {res && (
                        <span
                          className={cn(
                            "inline-flex max-w-[22rem] items-center gap-1 text-xs",
                            res.ok ? "text-[var(--success,#16a34a)]" : "text-[var(--danger,#dc2626)]"
                          )}
                          title={res.output}
                        >
                          {res.ok ? (
                            <CircleCheck className="h-3.5 w-3.5 shrink-0" />
                          ) : (
                            <CircleAlert className="h-3.5 w-3.5 shrink-0" />
                          )}
                          <span className="truncate">{res.ok ? t("conn.ok") : res.output}</span>
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
            {providers.length === 0 && (
              <tr>
                <td colSpan={5} className="px-3 py-4 text-center text-xs text-[var(--text-faint)]">
                  {t("conn.noConnections")}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="rounded-[var(--radius)] border border-[var(--border)] p-3">
        <div className="mb-2 text-xs font-medium text-[var(--text-muted)]">
          {editing ? t("conn.editConnection", { name: editing }) : t("conn.addConnection")}
        </div>
        <div className="flex flex-wrap items-end gap-2">
          <label className="text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">{t("conn.colName")}</span>
            <input
              aria-label={t("conn.providerNameAria")}
              value={name}
              disabled={editing !== null}
              placeholder="dashscope"
              onChange={(e) => setName(e.target.value)}
              className="w-40 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm outline-none focus:border-[var(--accent)] disabled:opacity-60"
            />
          </label>
          <label className="flex-1 text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">{t("conn.colBaseUrl")}</span>
            <input
              aria-label={t("conn.providerBaseUrlAria")}
              value={baseUrl}
              placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1"
              onChange={(e) => setBaseUrl(e.target.value)}
              className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 font-mono text-xs outline-none focus:border-[var(--accent)]"
            />
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">{t("conn.colApiKeyEnv")}</span>
            <input
              aria-label={t("conn.providerApiKeyEnvAria")}
              value={apiKeyEnv}
              placeholder="DASHSCOPE_API_KEY"
              onChange={(e) => setApiKeyEnv(e.target.value)}
              className="w-52 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 font-mono text-xs outline-none focus:border-[var(--accent)]"
            />
          </label>
          <button
            type="button"
            disabled={loading}
            onClick={submit}
            className="inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-2 text-sm font-medium text-[var(--accent-fg)] disabled:opacity-60"
          >
            <Plus className="h-4 w-4" />
            {editing ? t("common.save") : t("common.add")}
          </button>
          {editing && (
            <button
              type="button"
              onClick={reset}
              className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-2 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
            >
              {t("common.cancel")}
            </button>
          )}
        </div>
        <p className="mt-2 text-xs text-[var(--text-faint)]">
          {t("conn.foot.a")}
          <em>{t("conn.wordName")}</em>
          {t("conn.foot.b")}
          <code className="font-mono">.env</code>
          {t("conn.foot.c")}
        </p>
      </div>
    </section>
  );
}

// --------------------------------------------------------------------------- //
// Part 2 — Models
// --------------------------------------------------------------------------- //
function ModelsSection({
  providers,
  loading,
  cat,
  onSave,
}: {
  providers: ModelProviderDoc[];
  loading: boolean;
  cat: ModelCatalogDoc;
  onSave: (next: ModelCatalogDoc, onOk?: () => void) => Promise<void>;
}) {
  const { t } = useI18n();
  const [provider, setProvider] = useState("");
  const [type, setType] = useState<ModelType>("embedding");
  const [id, setId] = useState("");
  const [protocol, setProtocol] = useState<"openai" | "dashscope">("openai");
  const [dimension, setDimension] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [makeDefault, setMakeDefault] = useState(true);
  const [editing, setEditing] = useState<{ provider: string; id: string } | null>(null);

  const rows = providers.flatMap((p) =>
    (p.models ?? []).map((m) => ({ provider: p.name, model: m }))
  );

  const reset = () => {
    setProvider("");
    setType("embedding");
    setId("");
    setProtocol("openai");
    setDimension("");
    setBaseUrl("");
    setMakeDefault(true);
    setEditing(null);
  };

  const edit = (providerName: string, m: ModelSpecDoc) => {
    const t = (m.type ?? "chat") as ModelType;
    setEditing({ provider: providerName, id: m.id });
    setProvider(providerName);
    setType(t);
    setId(m.id);
    setProtocol(m.protocol === "dashscope" ? "dashscope" : "openai");
    setDimension(m.dimension != null ? String(m.dimension) : "");
    setBaseUrl(m.base_url ?? "");
    setMakeDefault(cat[defaultKeyFor(t)] === ref(providerName, m.id));
  };

  const submit = () => {
    const provName = provider.trim();
    const modelId = id.trim();
    if (!provName) {
      toast.error(t("conn.chooseConnection"));
      return;
    }
    if (!modelId) {
      toast.error(t("conn.modelIdRequired"));
      return;
    }
    const spec: ModelSpecDoc = { id: modelId, type };
    spec.protocol = protocol;
    if (type === "embedding" && dimension.trim()) {
      spec.dimension = Number(dimension) || undefined;
    }
    if (baseUrl.trim()) spec.base_url = baseUrl.trim();

    // Upsert the model into its connection's list by id, preserving the
    // connection's own base_url and any sibling models (e.g. a shared chat model).
    const origId = editing?.id;
    const list = (cat.providers ?? []).map((p) => {
      if (p.name !== provName) return p;
      const models = [...(p.models ?? [])];
      const i = models.findIndex((m) => m.id === (origId ?? modelId));
      if (i >= 0) models[i] = { ...models[i], ...spec };
      else models.push(spec);
      return { ...p, models };
    });
    let next: ModelCatalogDoc = { ...cat, providers: list };

    const newRef = ref(provName, modelId);
    const key = defaultKeyFor(type);
    if (makeDefault) {
      next[key] = newRef;
    } else if (editing && next[key] === ref(provName, origId ?? modelId)) {
      // Was the default for its type, now unchecked → clear it.
      next[key] = undefined;
    }
    // If an edit renamed the model id, repoint any stale default that still
    // referenced the old id.
    if (editing && origId && origId !== modelId) {
      next = clearDefaultsPointingAt(
        next,
        (r) => r === ref(provName, origId)
      );
      if (makeDefault) next[key] = newRef;
    }
    void onSave(next, reset);
  };

  const remove = (providerName: string, m: ModelSpecDoc) => {
    const list = (cat.providers ?? []).map((p) =>
      p.name === providerName
        ? { ...p, models: (p.models ?? []).filter((x) => x.id !== m.id) }
        : p
    );
    const gone = ref(providerName, m.id);
    const next = clearDefaultsPointingAt({ ...cat, providers: list }, (r) => r === gone);
    void onSave(next);
    if (editing && editing.provider === providerName && editing.id === m.id) reset();
  };

  const isDefault = (providerName: string, m: ModelSpecDoc) => {
    const t = (m.type ?? "chat") as ModelType;
    return cat[defaultKeyFor(t)] === ref(providerName, m.id);
  };

  return (
    <section className="space-y-3">
      <div>
        <h3 className="text-sm font-semibold">{t("conn.modelsTitle")}</h3>
        <p className="mt-1 text-xs text-[var(--text-muted)]">{t("conn.modelsHint")}</p>
      </div>

      <div className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)]">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--border)] text-left text-xs text-[var(--text-muted)]">
              <th className="px-3 py-2 font-medium">{t("conn.colConnection")}</th>
              <th className="px-3 py-2 font-medium">{t("conn.colModel")}</th>
              <th className="px-3 py-2 font-medium">{t("conn.colType")}</th>
              <th className="px-3 py-2 font-medium">{t("conn.colProtocol")}</th>
              <th className="px-3 py-2 font-medium">{t("conn.colDim")}</th>
              <th className="px-3 py-2" />
            </tr>
          </thead>
          <tbody>
            {rows.map(({ provider: pn, model: m }) => {
              const mt = (m.type ?? "chat") as ModelType;
              return (
                <tr key={`${pn}/${m.id}`} className="border-b border-[var(--border)] last:border-0">
                  <td className="px-3 py-2 text-[var(--text-muted)]">{pn}</td>
                  <td className="px-3 py-2 font-mono text-xs">
                    {m.id}
                    {isDefault(pn, m) && (
                      <span className="ml-2 rounded-full bg-[var(--accent)]/15 px-1.5 py-0.5 text-[10px] font-medium text-[var(--accent)]">
                        {t("conn.defaultBadge")}
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2">
                    <span className="rounded-full bg-[var(--surface-2)] px-2 py-0.5 text-xs">
                      {t(TYPE_LABEL_KEY[mt])}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-xs text-[var(--text-muted)]">
                    {m.protocol ?? "—"}
                  </td>
                  <td className="px-3 py-2 text-xs text-[var(--text-muted)]">
                    {m.dimension ?? "—"}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <div className="flex justify-end gap-2">
                      <button
                        type="button"
                        onClick={() => edit(pn, m)}
                        className="rounded-[var(--radius-sm)] border border-[var(--border)] px-2 py-1 text-xs hover:bg-[var(--surface-2)]"
                      >
                        {t("common.edit")}
                      </button>
                      <button
                        type="button"
                        aria-label={t("conn.deleteConnAria", { name: `${pn}/${m.id}` })}
                        onClick={() => remove(pn, m)}
                        className="rounded-[var(--radius-sm)] border border-[var(--border)] p-1.5 text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--danger)]"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
            {rows.length === 0 && (
              <tr>
                <td colSpan={6} className="px-3 py-4 text-center text-xs text-[var(--text-faint)]">
                  {t("conn.noModels")}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="rounded-[var(--radius)] border border-[var(--border)] p-3">
        <div className="mb-2 text-xs font-medium text-[var(--text-muted)]">
          {editing ? t("conn.editModel", { ref: `${editing.provider}/${editing.id}` }) : t("conn.registerModel")}
        </div>
        <div className="grid gap-3 md:grid-cols-3">
          <label className="text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">{t("conn.colConnection")}</span>
            <select
              aria-label={t("conn.modelProviderAria")}
              value={provider}
              disabled={editing !== null}
              onChange={(e) => setProvider(e.target.value)}
              className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm outline-none focus:border-[var(--accent)] disabled:opacity-60"
            >
              <option value="">{t("conn.selectPlaceholder")}</option>
              {providers.map((p) => (
                <option key={p.name} value={p.name}>{p.name}</option>
              ))}
            </select>
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">{t("conn.colType")}</span>
            <select
              aria-label={t("conn.modelTypeAria")}
              value={type}
              onChange={(e) => setType(e.target.value as ModelType)}
              className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
            >
              <option value="chat">{t("conn.typeOptChat")}</option>
              <option value="embedding">{t("conn.typeEmbedding")}</option>
              <option value="rerank">{t("conn.typeRerank")}</option>
            </select>
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">{t("conn.modelId")}</span>
            <input
              aria-label={t("conn.modelId")}
              value={id}
              placeholder="text-embedding-v4"
              onChange={(e) => setId(e.target.value)}
              className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 font-mono text-xs outline-none focus:border-[var(--accent)]"
            />
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">{t("conn.colProtocol")}</span>
            <select
              aria-label={t("conn.modelProtocolAria")}
              value={protocol}
              onChange={(e) => setProtocol(e.target.value as "openai" | "dashscope")}
              className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
            >
              <option value="openai">openai</option>
              <option value="dashscope">dashscope</option>
            </select>
          </label>
          {type === "embedding" && (
            <label className="text-sm">
              <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">{t("conn.dimension")}</span>
              <input
                aria-label={t("conn.embeddingDimAria")}
                value={dimension}
                type="number"
                min={1}
                placeholder="1024"
                onChange={(e) => setDimension(e.target.value)}
                className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
              />
            </label>
          )}
          <label className="text-sm">
            <span className="mb-1 block text-xs font-medium text-[var(--text-muted)]">{t("conn.baseUrlOverride")}</span>
            <input
              aria-label={t("conn.modelBaseUrlAria")}
              value={baseUrl}
              placeholder={t("conn.baseUrlPlaceholder")}
              onChange={(e) => setBaseUrl(e.target.value)}
              className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 font-mono text-xs outline-none focus:border-[var(--accent)]"
            />
          </label>
        </div>
        <div className="mt-3 flex items-center justify-between gap-3">
          <label className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
            <input
              aria-label={t("conn.setDefaultAria")}
              type="checkbox"
              checked={makeDefault}
              onChange={(e) => setMakeDefault(e.target.checked)}
              className="h-4 w-4"
            />
            {t("conn.setDefault", { type: t(TYPE_LABEL_KEY[type]) })}
          </label>
          <div className="flex gap-2">
            {editing && (
              <button
                type="button"
                onClick={reset}
                className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-2 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)]"
              >
                {t("common.cancel")}
              </button>
            )}
            <button
              type="button"
              disabled={loading}
              onClick={submit}
              className="inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-2 text-sm font-medium text-[var(--accent-fg)] disabled:opacity-60"
            >
              <Plus className="h-4 w-4" />
              {editing ? t("common.save") : t("conn.register")}
            </button>
          </div>
        </div>
        {providers.length === 0 && (
          <p className="mt-2 text-xs text-[var(--text-faint)]">
            {t("conn.addConnFirst")}
          </p>
        )}
      </div>
    </section>
  );
}
