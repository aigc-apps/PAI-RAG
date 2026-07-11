import { Database, RefreshCw } from "lucide-react";
import type {
  AgentConfigDocument,
  ModelCatalogDoc,
  ModelProviderDoc,
} from "../api/agentConfig";
import { EngineStatusBadge, useEngineStatus } from "./EngineStatus";
import { cn } from "../lib/cn";
import { useI18n, type TFunction } from "../i18n";

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
  const { t } = useI18n();
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
        <h2 className="text-xl font-semibold tracking-tight">{t("kb.title")}</h2>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--text-muted)]">
          {t("kb.introA")}
          <strong>Models</strong>
          {t("kb.introB")}
        </p>
      </div>

      {/* Vector database — configurable */}
      <section className="space-y-3">
        <div>
          <h3 className="text-sm font-semibold">{t("kb.vectorDb")}</h3>
          <p className="mt-1 text-xs text-[var(--text-muted)]">{t("kb.vectorDbHint")}</p>
        </div>
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-[var(--radius-sm)] bg-[var(--surface-2)] p-2 text-[var(--text-muted)]">
              <Database className="h-4 w-4" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="text-sm font-medium">{t("kb.vectorStore")}</div>
              <div className="truncate font-mono text-xs text-[var(--text-muted)]">
                {configured ? vdb.url : t("engine.unconfigured")}
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
                title={t("kb.recheck")}
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
              {t("kb.configure")}
            </button>
          </div>
          {/* Show the reason inline (not just a tooltip) when unreachable or
              when config is incomplete, so the fix is obvious on the page. */}
          {configured && engine && engine.configured && !engine.healthy && (
            <p className="mt-3 border-t border-[var(--border)] pt-3 text-xs text-[var(--warning,#d97706)]">
              {engine.detail || t("kb.esUnreachableInline")}
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
          <h3 className="text-sm font-semibold">{t("kb.retrievalModels")}</h3>
          <p className="mt-1 text-xs text-[var(--text-muted)]">
            {t("kb.retrievalModelsHintA")}
            <strong>Models</strong>
            {t("kb.retrievalModelsHintB")}
          </p>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <RagModelReadout
            t={t}
            label={t("kb.embeddingModel")}
            defaultRef={cat.default_embedding_model}
            models={modelsByType("embedding")}
            emptyHint={t("kb.embeddingEmpty")}
          />
          <RagModelReadout
            t={t}
            label={t("kb.rerankModel")}
            defaultRef={cat.default_rerank_model}
            models={modelsByType("rerank")}
            emptyHint={t("kb.rerankEmpty")}
          />
        </div>
      </section>
    </div>
  );
}

/** Read-only summary of the models of one RAG role: which is the default and
 * what else is available. Registration happens on the Models tab. */
function RagModelReadout({
  t,
  label,
  defaultRef,
  models,
  emptyHint,
}: {
  t: TFunction;
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
          {t("kb.default")}
          {defaultRef ? (
            <code className="font-mono">{defaultRef}</code>
          ) : (
            <span className="text-[var(--text-faint)]">{t("kb.notSet")}</span>
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
