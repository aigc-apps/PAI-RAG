import { Database, RefreshCw } from "lucide-react";
import type {
  AgentConfigDocument,
  ModelCatalogDoc,
  ModelProviderDoc,
} from "../api/agentConfig";
import { EngineStatusBadge, useEngineStatus } from "./EngineStatus";
import { cn } from "../lib/cn";

// A model's type as the backend `ModelSpec` stores it.
type ModelType = "chat" | "embedding" | "rerank";

/** Reference format used everywhere the catalog points at a model. */
const ref = (provider: string, id: string) => `${provider}/${id}`;

const providersOf = (cat: ModelCatalogDoc): ModelProviderDoc[] =>
  cat.providers ?? [];

/**
 * Knowledge Base settings: the RAG components used for retrieval. The vector
 * database is configurable here (reusing the existing dialog); the embedding and
 * rerank models are read-only summaries — they are registered on the Models tab
 * and surfaced here so you can see what retrieval will use.
 */
export function KnowledgeBasePanel({
  doc,
  onConfigureVectorDB,
}: {
  doc: AgentConfigDocument;
  onConfigureVectorDB: () => void;
}) {
  const cat = doc.models ?? {};
  const vdb = doc.knowledgebase.vectordb;
  const configured = vdb.engine === "elasticsearch" && !!vdb.url;

  // Live reachability: the config-graded `vdb.status` only tells us the URL +
  // secret are filled in, not that ES actually answers. Probe the live engine
  // (GET /v1/knowledge/engine → ES ping) and reflect the real state here. Gated
  // on `configured` (skip the request for the always-reachable local engine) and
  // re-probed when the saved URL changes (a save rebuilds the backend engine).
  const { engine, checking, refresh } = useEngineStatus(configured, vdb.url);

  const providers = providersOf(cat);
  const modelsByType = (type: ModelType) =>
    providers.flatMap((p) =>
      (p.models ?? [])
        .filter((m) => (m.type ?? "chat") === type)
        .map((m) => ref(p.name, m.id))
    );

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold tracking-tight">Knowledge Base</h2>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--text-muted)]">
          The components used for RAG retrieval. Configure the vector database
          here; the embedding and rerank models are registered on the{" "}
          <strong>Models</strong> tab and shown read-only. A new knowledge base
          inherits whichever vector engine is active here.
        </p>
      </div>

      {/* Vector database — configurable */}
      <section className="space-y-3">
        <div>
          <h3 className="text-sm font-semibold">向量数据库</h3>
          <p className="mt-1 text-xs text-[var(--text-muted)]">
            知识库检索使用 Elasticsearch，全局生效。保存后新建的知识库即记录该引擎。
          </p>
        </div>
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-[var(--radius-sm)] bg-[var(--surface-2)] p-2 text-[var(--text-muted)]">
              <Database className="h-4 w-4" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="text-sm font-medium">向量库 · Elasticsearch</div>
              <div className="truncate font-mono text-xs text-[var(--text-muted)]">
                {configured ? vdb.url : "未配置"}
              </div>
            </div>
            <EngineStatusBadge
              variant="dot"
              configured={configured}
              checking={checking}
              engine={engine}
            />
            {configured && (
              <button
                type="button"
                onClick={() => void refresh()}
                disabled={checking}
                title="重新检测可达性"
                className="rounded-[var(--radius-sm)] border border-[var(--border)] p-1.5 text-[var(--text-muted)] hover:bg-[var(--surface-2)] disabled:opacity-50"
              >
                <RefreshCw className={cn("h-3.5 w-3.5", checking && "animate-spin")} />
              </button>
            )}
            <button
              type="button"
              onClick={onConfigureVectorDB}
              className="rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-1.5 text-xs hover:bg-[var(--surface-2)]"
            >
              Configure
            </button>
          </div>
          {/* Show the reason inline (not just a tooltip) when unreachable or
              when config is incomplete, so the fix is obvious on the page. */}
          {configured && engine && engine.configured && !engine.healthy && (
            <p className="mt-3 border-t border-[var(--border)] pt-3 text-xs text-[var(--warning,#d97706)]">
              {engine.detail || "已配置 Elasticsearch 但当前不可达，检索将自动降级为本地引擎。"}
            </p>
          )}
          {!configured && vdb.error && (
            <p className="mt-3 border-t border-[var(--border)] pt-3 text-xs text-[var(--text-faint)]">
              {vdb.error}
            </p>
          )}
        </div>
      </section>

      {/* Embedding + rerank — read-only, registered on the Models tab */}
      <section className="space-y-3">
        <div>
          <h3 className="text-sm font-semibold">检索模型</h3>
          <p className="mt-1 text-xs text-[var(--text-muted)]">
            只读。在 <strong>Models</strong> 标签页注册与设置默认模型。
          </p>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <RagModelReadout
            label="Embedding 模型"
            defaultRef={cat.default_embedding_model}
            models={modelsByType("embedding")}
            emptyHint="在 Models 中注册一个 embedding 模型"
          />
          <RagModelReadout
            label="Rerank 模型"
            defaultRef={cat.default_rerank_model}
            models={modelsByType("rerank")}
            emptyHint="在 Models 中注册一个 rerank 模型（可选）"
          />
        </div>
      </section>
    </div>
  );
}

/** Read-only summary of the models of one RAG role: which is the default and
 * what else is available. Registration happens on the Models tab. */
function RagModelReadout({
  label,
  defaultRef,
  models,
  emptyHint,
}: {
  label: string;
  defaultRef?: string;
  models: string[];
  emptyHint: string;
}) {
  return (
    <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="flex items-center justify-between gap-2">
        <div className="text-sm font-medium">{label}</div>
        <div className="text-xs text-[var(--text-muted)]">
          默认：
          {defaultRef ? (
            <code className="font-mono">{defaultRef}</code>
          ) : (
            <span className="text-[var(--text-faint)]">未设置</span>
          )}
        </div>
      </div>
      {models.length ? (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {models.map((r) => {
            const isDefault = r === defaultRef;
            return (
              <span
                key={r}
                className={cn(
                  "rounded-full px-2 py-0.5 font-mono text-[11px]",
                  isDefault
                    ? "bg-[var(--accent)]/15 text-[var(--accent)]"
                    : "bg-[var(--surface-2)] text-[var(--text-muted)]"
                )}
              >
                {r}
              </span>
            );
          })}
        </div>
      ) : (
        <div className="mt-3 text-xs text-[var(--text-faint)]">{emptyHint}</div>
      )}
    </div>
  );
}
