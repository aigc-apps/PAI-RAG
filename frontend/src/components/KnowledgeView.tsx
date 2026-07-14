import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  BookOpen, Bot, Check, ChevronDown, Copy, Database, Eye, Globe, Info, Loader2, Pencil, Plus,
  RefreshCw, Search, Trash2, TriangleAlert, Upload, X,
} from "lucide-react";
import { toast } from "sonner";
import {
  cancelDataSourceSync, createDataSource, createKnowledgeBase, deleteDataSource, deleteKnowledgeBase,
  getUploadSupport, importKnowledgeDocument, listDataSources,
  listKnowledgeBases, listKnowledgeChunks, listKnowledgeDocuments, searchKnowledge,
  syncDataSource, updateDataSource, updateKnowledgeBase, uploadKnowledgeDocument,
  type KnowledgeBase, type KnowledgeBasePatch, type KnowledgeChunk,
  type KnowledgeDataSource, type KnowledgeDocument, type KnowledgeHit,
  type KnowledgeSyncProgress,
} from "../api/knowledge";
import { listModels, modelsByType, type ModelInfo } from "../api/models";
import { EngineStatusBadge, useEngineStatus } from "./EngineStatus";
import { cn } from "../lib/cn";
import { CARD, INPUT, BTN_PRIMARY, BTN_GHOST, ICON_BTN, PILL_BASE } from "../lib/ui";
import { PageHeader } from "./PageHeader";
import { copyText } from "../lib/clipboard";
import { useI18n, type TFunction, type MessageKey } from "../i18n";

// --- config accessors with defaults ---------------------------------------- //
const num = (v: unknown, d: number) => (typeof v === "number" && !Number.isNaN(v) ? v : d);
const str = (v: unknown, d: string) => (typeof v === "string" && v ? v : d);
const parserOf = (kb: KnowledgeBase) => ({
  chunk_size: num(kb.default_parser_config?.chunk_size, 1000),
  chunk_overlap: num(kb.default_parser_config?.chunk_overlap, 150),
});
const retrievalOf = (kb: KnowledgeBase) => ({
  mode: (kb.default_retrieval_config?.mode ?? "hybrid") as "hybrid" | "vector" | "keyword",
  top_k: num(kb.default_retrieval_config?.top_k, 6),
  score_threshold: num(kb.default_retrieval_config?.score_threshold, 0),
  force_citation: kb.default_retrieval_config?.force_citation ?? true,
});

// Load the embedding model catalog once for the create picker.
// Empty lists (no models configured) → callers fall back to the local defaults.
interface ModelCatalog {
  embedding: ModelInfo[];
  defaultEmbedding: string | null;
}
function useModelCatalog(enabled: boolean): ModelCatalog {
  const [cat, setCat] = useState<ModelCatalog>({
    embedding: [], defaultEmbedding: null,
  });
  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    listModels()
      .then((c) => {
        if (!alive) return;
        setCat({
          embedding: modelsByType(c, "embedding"),
          defaultEmbedding: c.defaultEmbedding,
        });
      })
      .catch(() => { /* keep empty → local defaults */ });
    return () => { alive = false; };
  }, [enabled]);
  return cat;
}

// Status/source-type values that carry a localized label. Unlisted values
// (website / upload / text / file) render as their raw identifier in both langs.
const STATUS_KEY: Record<string, MessageKey> = {
  ready: "kbview.status.ready", empty: "kbview.status.empty", indexed: "kbview.status.indexed",
  active: "kbview.status.active", processing: "kbview.status.processing", failed: "kbview.status.failed",
  has_errors: "kbview.status.has_errors", disabled: "kbview.status.disabled", deleted: "kbview.status.deleted",
  idle: "kbview.status.idle", syncing: "kbview.status.syncing", succeeded: "kbview.status.succeeded",
  partial: "kbview.status.partial", llms_txt: "kbview.status.llms_txt", yuque: "kbview.status.yuque",
};

function statusLabelOf(t: TFunction, status: string): string {
  return STATUS_KEY[status] ? t(STATUS_KEY[status]) : status;
}

function statusClass(status: string) {
  if (["ready", "indexed", "active", "succeeded"].includes(status))
    return "border-[var(--success)]/35 bg-[var(--success)]/10 text-[var(--success)]";
  if (["failed", "has_errors"].includes(status))
    return "border-[var(--danger)]/35 bg-[var(--danger)]/10 text-[var(--danger)]";
  if (status === "partial")
    return "border-[var(--warning)]/40 bg-[var(--warning)]/10 text-[var(--warning)]";
  if (status === "syncing" || status === "processing")
    return "border-[var(--accent)]/35 bg-[var(--accent-soft)] text-[var(--accent)]";
  return "border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)]";
}

function Pill({ status }: { status: string }) {
  const { t } = useI18n();
  return (
    <span className={cn(PILL_BASE, statusClass(status))}>
      {statusLabelOf(t, status)}
    </span>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-medium text-[var(--text-muted)]">{label}</span>
      {children}
      {hint && <span className="mt-1.5 block text-[11px] text-[var(--text-faint)]">{hint}</span>}
    </label>
  );
}

function Toggle({ on, onChange, label, disabled }: {
  on: boolean; onChange?: (v: boolean) => void; label: string; disabled?: boolean;
}) {
  return (
    <button
      type="button" disabled={disabled}
      onClick={() => onChange?.(!on)}
      className={cn("inline-flex items-center gap-2.5 text-sm", disabled && "cursor-not-allowed opacity-50")}
    >
      <span className={cn(
        "relative h-5 w-9 flex-shrink-0 rounded-full transition-colors",
        on ? "bg-[var(--accent)]" : "bg-[var(--surface-3)]"
      )}>
        <span className={cn(
          "absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-white shadow-[var(--shadow-sm)] transition-transform",
          on && "translate-x-4"
        )} />
      </span>
      <span className="text-[var(--text)]">{label}</span>
    </button>
  );
}

function Seg<T extends string>({ value, options, onChange }: {
  value: T; options: { value: T; label: string }[]; onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex overflow-hidden rounded-[var(--radius-sm)] border border-[var(--border)]">
      {options.map((o, i) => (
        <button
          key={o.value} type="button" onClick={() => onChange(o.value)}
          className={cn(
            "px-3.5 py-1.5 text-[13px] font-medium",
            i > 0 && "border-l border-[var(--border)]",
            value === o.value
              ? "bg-[var(--accent-soft)] text-[var(--accent)]"
              : "bg-[var(--bg)] text-[var(--text-muted)] hover:text-[var(--text)]"
          )}
        >{o.label}</button>
      ))}
    </div>
  );
}

function Drawer({ open, title, onClose, footer, children }: {
  open: boolean; title: string; onClose: () => void; footer: ReactNode; children: ReactNode;
}) {
  const { t } = useI18n();
  return (
    <>
      <div
        onClick={onClose}
        className={cn("fixed inset-0 z-40 bg-black/30 transition-opacity",
          open ? "opacity-100" : "pointer-events-none opacity-0")}
      />
      <div className={cn(
        "fixed top-0 right-0 z-50 flex h-full w-[min(460px,94vw)] flex-col border-l border-[var(--border)] bg-[var(--bg)] shadow-[var(--shadow)] transition-transform",
        open ? "translate-x-0" : "translate-x-full"
      )}>
        <div className="flex items-center gap-2 border-b border-[var(--border)] px-4 py-3">
          <h3 className="text-sm font-semibold">{title}</h3>
          <div className="flex-1" />
          <button type="button" aria-label={t("common.close")} onClick={onClose} className={ICON_BTN}>
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="flex-1 space-y-3.5 overflow-y-auto p-4 scrollbar-thin">{children}</div>
        <div className="flex justify-end gap-2 border-t border-[var(--border)] px-4 py-3">{footer}</div>
      </div>
    </>
  );
}

// ======================================================================== //
// Root
// ======================================================================== //
export type KnowledgeTab = "overview" | "config" | "datasources" | "files" | "recall";

export function KnowledgeView({
  onBack,
  kbId,
  tab,
  onOpenKb,
  onBackToList,
  onTabChange,
  onInvalidKb,
}: {
  onBack: () => void;
  kbId?: string;
  tab: KnowledgeTab;
  onOpenKb: (id: string) => void;
  onBackToList: () => void;
  onTabChange: (tab: KnowledgeTab) => void;
  onInvalidKb: () => void;
}) {
  const { t } = useI18n();
  const [bases, setBases] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = async () => {
    const rows = await listKnowledgeBases();
    setBases(rows);
    return rows;
  };

  useEffect(() => {
    (async () => {
      setLoading(true);
      try { await refresh(); }
      catch (e) { toast.error(e instanceof Error ? e.message : t("kbview.loadKbFailed")); }
      finally { setLoading(false); }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const open = kbId ? bases.find((b) => b.id === kbId) ?? null : null;

  useEffect(() => {
    if (!loading && kbId && !open) onInvalidKb();
  }, [kbId, loading, onInvalidKb, open]);

  if (open) {
    return (
      <KbDetail
        kb={open}
        tab={tab}
        onTabChange={onTabChange}
        onBackToList={onBackToList}
        onChanged={async () => { await refresh(); }}
        onDeleted={async () => { onBackToList(); await refresh(); }}
      />
    );
  }

  return (
    <div className="workspace-page flex h-full flex-col text-[var(--text)]">
      <TopBar onBack={onBack} crumb={<b className="font-semibold text-[var(--text)]">{t("kb.title")}</b>} />
      <div className="min-h-0 flex-1 overflow-y-auto scrollbar-thin">
        <KbList
          bases={bases} loading={loading}
          onOpen={onOpenKb}
          onCreated={async (id) => { await refresh(); onOpenKb(id); }}
          onChanged={refresh}
        />
      </div>
    </div>
  );
}

function TopBar({ onBack, crumb }: { onBack: () => void; crumb: ReactNode }) {
  const { t } = useI18n();
  return <PageHeader icon={Database} title={crumb} onBack={onBack} backLabel={t("common.back")} />;
}

// ======================================================================== //
// List
// ======================================================================== //
function KbList({ bases, loading, onOpen, onCreated, onChanged }: {
  bases: KnowledgeBase[]; loading: boolean;
  onOpen: (id: string) => void; onCreated: (id: string) => void;
  onChanged: () => Promise<KnowledgeBase[]>;
}) {
  const { t } = useI18n();
  const [showCreate, setShowCreate] = useState(false);
  const [editing, setEditing] = useState<KnowledgeBase | null>(null);

  const remove = async (kb: KnowledgeBase) => {
    if (!confirm(t("kbview.confirmDeleteKb", { name: kb.name }))) return;
    try {
      await deleteKnowledgeBase(kb.id);
      toast.success(t("kbview.deleted"));
      await onChanged();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : t("kbview.deleteFailed"));
    }
  };

  return (
    <div className="workspace-container">
      <div className="mb-7 grid max-w-5xl grid-cols-[minmax(0,1fr)_auto] items-start gap-x-8 gap-y-4 max-sm:grid-cols-1">
        <div className="min-w-0">
          <h1 className="text-[22px] font-semibold tracking-tight">{t("kb.title")}</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--text-muted)]">
            {t("kbview.listIntro")}
          </p>
        </div>
        <button
          type="button"
          className={cn(BTN_PRIMARY, "mt-2 whitespace-nowrap max-sm:mt-0 max-sm:w-fit")}
          aria-label={t("kbview.newKb")}
          title={t("kbview.newKb")}
          onClick={() => setShowCreate(true)}
        >
          <Plus className="h-4 w-4" /> {t("kbview.newShort")}
        </button>
      </div>

      {loading ? (
        <div className="flex items-center gap-2 py-10 text-sm text-[var(--text-muted)]">
          <Loader2 className="h-4 w-4 animate-spin" /> {t("common.loading")}
        </div>
      ) : bases.length === 0 ? (
        <div className="grid place-items-center rounded-[var(--radius-lg)] border border-dashed border-[var(--border)] py-16 text-center">
          <Database className="h-6 w-6 text-[var(--text-faint)]" />
          <div className="mt-3 text-sm text-[var(--text-muted)]">{t("kbview.emptyKb")}</div>
          <button className={cn(BTN_GHOST, "mt-3")} onClick={() => setShowCreate(true)}>
            <Plus className="h-4 w-4" /> {t("kbview.createFirst")}
          </button>
        </div>
      ) : (
        <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fill,minmax(320px,1fr))" }}>
          {bases.map((kb) => (
            <KbCard
              key={kb.id} kb={kb}
              onOpen={() => onOpen(kb.id)}
              onEdit={() => setEditing(kb)}
              onDelete={() => void remove(kb)}
            />
          ))}
        </div>
      )}

      <CreateDrawer
        open={showCreate}
        onClose={() => setShowCreate(false)}
        onCreated={(id) => { setShowCreate(false); onCreated(id); }}
      />
      <EditKbDrawer
        kb={editing}
        onClose={() => setEditing(null)}
        onSaved={async () => { setEditing(null); await onChanged(); }}
      />
    </div>
  );
}

function KbCard({ kb, onOpen, onEdit, onDelete }: {
  kb: KnowledgeBase; onOpen: () => void; onEdit: () => void; onDelete: () => void;
}) {
  const { t } = useI18n();
  const emb = kb.embedding_config ?? {};
  const parser = parserOf(kb);
  return (
    <div className="professional-card professional-card-hover relative flex min-h-[170px] flex-col gap-3 p-[18px] text-left">
      <div className="flex items-start justify-between gap-2">
        <button type="button" onClick={onOpen} className="focus-ring min-w-0 flex-1 rounded text-left">
          <h3 className="truncate text-[15px] font-semibold">{kb.name}</h3>
        </button>
        <div className="flex items-center gap-1">
          <Pill status={kb.status} />
          <button type="button" className={ICON_BTN} title={t("common.edit")} aria-label={`${t("common.edit")} ${kb.name}`} onClick={onEdit}>
            <Pencil className="h-4 w-4" />
          </button>
          <button type="button" className={cn(ICON_BTN, "hover:text-[var(--danger)]")} title={t("common.delete")} aria-label={`${t("common.delete")} ${kb.name}`} onClick={onDelete}>
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>
      <button type="button" onClick={onOpen} className="focus-ring flex flex-1 flex-col gap-3 rounded text-left">
        <div className="min-h-[20px] truncate text-[13px] text-[var(--text-muted)]">
          {kb.description || t("kbview.noDescription")}
        </div>
      <div className="flex items-center gap-4 border-y border-[var(--border)] py-2.5 text-xs text-[var(--text-muted)]">
        <span><b className="font-semibold text-[var(--text)]">{kb.document_count}</b> {t("kbview.docs")}</span>
        <span><b className="font-semibold text-[var(--text)]">{kb.chunk_count}</b> {t("kbview.chunks")}</span>
        <span className="rounded-full border border-[var(--border)] bg-[var(--surface)] px-2 py-0.5">{kb.visibility}</span>
      </div>
      <div className="flex flex-wrap gap-1.5 pt-0.5">
        <Chip mono>{str(emb.model, "local-hash-v1")} · {num(emb.dimension, 64)}d</Chip>
        <Chip>{t("kbview.chipVector")} <span className="text-[var(--text-faint)]">{str(kb.vector_store_config?.provider_id, "local_sql")}</span></Chip>
        <Chip>{t("kbview.chipChunk")} <span className="text-[var(--text-faint)]">{parser.chunk_size}/{parser.chunk_overlap}</span></Chip>
      </div>
      </button>
    </div>
  );
}

function EditKbDrawer({ kb, onClose, onSaved }: {
  kb: KnowledgeBase | null; onClose: () => void; onSaved: () => Promise<void>;
}) {
  const { t } = useI18n();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [visibility, setVisibility] = useState("private");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!kb) return;
    setName(kb.name);
    setDescription(kb.description ?? "");
    setVisibility(kb.visibility);
  }, [kb]);

  const save = async () => {
    if (!kb || !name.trim()) return;
    setSaving(true);
    try {
      await updateKnowledgeBase(kb.id, { name: name.trim(), description: description.trim(), visibility });
      toast.success(t("kbview.configSaved"));
      await onSaved();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : t("kbview.saveFailed"));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Drawer open={!!kb} title={t("kbview.editKb")} onClose={onClose} footer={<>
      <button type="button" className={BTN_GHOST} onClick={onClose}>{t("common.cancel")}</button>
      <button type="button" className={BTN_PRIMARY} disabled={saving || !name.trim()} onClick={() => void save()}>
        {saving && <Loader2 className="h-4 w-4 animate-spin" />} {t("common.save")}
      </button>
    </>}>
      <Field label={t("kbview.name")}><input className={INPUT} value={name} onChange={(e) => setName(e.target.value)} /></Field>
      <Field label={t("kbview.description")}><input className={INPUT} value={description} onChange={(e) => setDescription(e.target.value)} placeholder={t("common.optional")} /></Field>
      <Field label={t("kbview.visibility")}>
        <select className={INPUT} value={visibility} onChange={(e) => setVisibility(e.target.value)}>
          <option value="private">{t("kbview.visPrivate")}</option>
          <option value="workspace">{t("kbview.visWorkspace")}</option>
          <option value="public">public</option>
        </select>
      </Field>
    </Drawer>
  );
}

function Chip({ children, mono }: { children: ReactNode; mono?: boolean }) {
  return (
    <span className={cn(
      "inline-flex items-center gap-1 rounded-full border border-[var(--border)] bg-[var(--surface)] px-2 py-0.5 text-[11px] text-[var(--text-muted)]",
      mono && "font-mono text-[10.5px]"
    )}>{children}</span>
  );
}

// ======================================================================== //
// Detail
// ======================================================================== //
function KbDetail({ kb, tab, onTabChange, onBackToList, onChanged, onDeleted }: {
  kb: KnowledgeBase; onBackToList: () => void;
  tab: KnowledgeTab; onTabChange: (tab: KnowledgeTab) => void;
  onChanged: () => Promise<void>; onDeleted: () => Promise<void>;
}) {
  const { t } = useI18n();

  const tabs: { id: KnowledgeTab; label: string; badge?: number }[] = [
    { id: "overview", label: t("kbview.tab.overview") },
    { id: "config", label: t("kbview.tab.config") },
    { id: "datasources", label: t("kbview.tab.datasources") },
    { id: "files", label: t("kbview.tab.files"), badge: kb.document_count },
    { id: "recall", label: t("kbview.tab.recall") },
  ];

  const remove = async () => {
    if (!confirm(t("kbview.confirmDeleteKb", { name: kb.name }))) return;
    try { await deleteKnowledgeBase(kb.id); toast.success(t("kbview.deleted")); await onDeleted(); }
    catch (e) { toast.error(e instanceof Error ? e.message : t("kbview.deleteFailed")); }
  };

  return (
    <div className="workspace-page flex h-full flex-col text-[var(--text)]">
      <TopBar
        onBack={onBackToList}
        crumb={<>
          <span className="cursor-pointer hover:text-[var(--text)]" onClick={onBackToList}>{t("kb.title")}</span>
          <span className="mx-1.5 text-[var(--text-faint)]">/</span>
          <b className="font-semibold text-[var(--text)]">{kb.name}</b>
        </>}
      />
      <div className="flex flex-shrink-0 items-center gap-1 border-b border-[var(--border)] bg-[var(--bg-elevated)]/84 px-4 backdrop-blur">
        {tabs.map((tb) => (
          <button
            key={tb.id} type="button" onClick={() => onTabChange(tb.id)}
            className={cn(
              "-mb-px border-b-2 px-4 py-2.5 text-sm font-medium transition-colors",
              tab === tb.id
                ? "border-[var(--accent)] text-[var(--text)]"
                : "border-transparent text-[var(--text-muted)] hover:text-[var(--text)]"
            )}
          >
            {tb.label}{typeof tb.badge === "number" && <span className="ml-1 text-[var(--text-faint)]">{tb.badge}</span>}
          </button>
        ))}
        <div className="flex-1" />
        <button type="button" onClick={remove} className={cn(ICON_BTN, "hover:text-[var(--danger)]")} title={t("kbview.deleteKb")}>
          <Trash2 className="h-4 w-4" />
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto scrollbar-thin">
        <div className="workspace-container">
          {tab === "overview" && <OverviewPanel kb={kb} />}
          {tab === "config" && <ConfigPanel kb={kb} onSaved={onChanged} />}
          {tab === "datasources" && <DataSourcePanel kb={kb} onChanged={onChanged} />}
          {tab === "files" && <FilesPanel kb={kb} onChanged={onChanged} />}
          {tab === "recall" && <RecallPanel kb={kb} />}
        </div>
      </div>
    </div>
  );
}

function Stat({ n, l }: { n: ReactNode; l: string }) {
  return (
    <div className="professional-card px-5 py-4">
      <div className="text-[22px] font-semibold tracking-tight">{n}</div>
      <div className="mt-1 text-xs font-medium text-[var(--text-faint)]">{l}</div>
    </div>
  );
}

function KV({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="flex justify-between gap-3 border-b border-[var(--border)] py-2 text-sm last:border-b-0">
      <span className="text-[var(--text-muted)]">{k}</span>
      <span className="font-mono text-xs">{v}</span>
    </div>
  );
}

function OverviewPanel({ kb }: { kb: KnowledgeBase }) {
  const { t } = useI18n();
  const emb = kb.embedding_config ?? {};
  const parser = parserOf(kb);
  const ret = retrievalOf(kb);
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <Stat n={kb.document_count} l={t("kbview.docs")} />
        <Stat n={kb.chunk_count} l={t("kbview.chunksActive")} />
        <Stat n={statusLabelOf(t, kb.status)} l={t("kbview.statusLabel")} />
        <Stat n={kb.visibility} l={t("kbview.visibility")} />
      </div>
      <div className={CARD}>
        <div className="mb-1 flex items-center gap-2">
          <h3 className="text-[15px] font-semibold">{t("kbview.configSummary")}</h3>
          <span className="ml-auto inline-flex items-center gap-1.5 text-[11px] text-[var(--text-muted)]">
            <Info className="h-3 w-3" /> {t("kbview.modelsFromProviders")}
          </span>
        </div>
        <div className="mt-4 grid gap-x-8 md:grid-cols-2">
          <div>
            <KV k="Embedding" v={`${str(emb.provider_id, "local_hash")} / ${str(emb.model, "local-hash-v1")}`} />
            <KV k={t("kbview.vectorDim")} v={num(emb.dimension, 64)} />
            <KV k={t("kbview.vectorEngine")} v={`${str(kb.vector_store_config?.provider_id, "local_sql")} · ${str(kb.vector_store_config?.metric, "cosine")}`} />
          </div>
          <div>
            <KV k={t("kbview.chunkingLabel")} v={`${parser.chunk_size} / overlap ${parser.chunk_overlap}`} />
            <KV k={t("kbview.retrievalMode")} v={`${ret.mode} · top_k ${ret.top_k}`} />
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------- Config ---------- //
function ConfigPanel({ kb, onSaved }: { kb: KnowledgeBase; onSaved: () => Promise<void> }) {
  const { t } = useI18n();
  const initParser = parserOf(kb), initRet = retrievalOf(kb);
  const [name, setName] = useState(kb.name);
  const [description, setDescription] = useState(kb.description ?? "");
  const [visibility, setVisibility] = useState(kb.visibility);
  const [chunkSize, setChunkSize] = useState(initParser.chunk_size);
  const [chunkOverlap, setChunkOverlap] = useState(initParser.chunk_overlap);
  const [mode, setMode] = useState(initRet.mode);
  const [topK, setTopK] = useState(initRet.top_k);
  const [threshold, setThreshold] = useState(initRet.score_threshold);
  const [forceCite, setForceCite] = useState(initRet.force_citation);
  const [saving, setSaving] = useState(false);

  const emb = kb.embedding_config ?? {};
  const vec = kb.vector_store_config ?? {};

  const dirty =
    name !== kb.name || description !== (kb.description ?? "") || visibility !== kb.visibility ||
    chunkSize !== initParser.chunk_size || chunkOverlap !== initParser.chunk_overlap ||
    mode !== initRet.mode || topK !== initRet.top_k ||
    threshold !== initRet.score_threshold || forceCite !== initRet.force_citation;
  const indexAffecting = chunkSize !== initParser.chunk_size || chunkOverlap !== initParser.chunk_overlap;

  const reset = () => {
    setName(kb.name); setDescription(kb.description ?? ""); setVisibility(kb.visibility);
    setChunkSize(initParser.chunk_size); setChunkOverlap(initParser.chunk_overlap);
    setMode(initRet.mode); setTopK(initRet.top_k);
    setThreshold(initRet.score_threshold); setForceCite(initRet.force_citation);
  };

  const save = async () => {
    setSaving(true);
    try {
      const patch: KnowledgeBasePatch = {
        name, description, visibility,
        default_parser_config: { chunk_size: chunkSize, chunk_overlap: chunkOverlap },
        default_retrieval_config: { mode, top_k: topK, score_threshold: threshold, force_citation: forceCite },
      };
      await updateKnowledgeBase(kb.id, patch);
      toast.success(t("kbview.configSaved"));
      await onSaved();
    } catch (e) { toast.error(e instanceof Error ? e.message : t("kbview.saveFailed")); }
    finally { setSaving(false); }
  };

  return (
    <div className="space-y-3.5">
      {/* basic */}
      <div className={CARD}>
        <h3 className="text-sm font-semibold">{t("kbview.basicInfo")}</h3>
        <div className="mt-3 grid gap-3.5 md:grid-cols-2">
          <Field label={t("kbview.name")}><input className={INPUT} value={name} onChange={(e) => setName(e.target.value)} /></Field>
          <Field label={t("kbview.visibility")}>
            <select className={INPUT} value={visibility} onChange={(e) => setVisibility(e.target.value)}>
              <option value="private">{t("kbview.visPrivate")}</option>
              <option value="workspace">{t("kbview.visWorkspace")}</option>
              <option value="public">public</option>
            </select>
          </Field>
          <div className="md:col-span-2">
            <Field label={t("kbview.description")}><input className={INPUT} value={description} onChange={(e) => setDescription(e.target.value)} placeholder={t("common.optional")} /></Field>
          </div>
        </div>
      </div>

      {/* Embedding (frozen at creation) + vector engine — read-only */}
      <div className={CARD}>
        <div className="mb-1 flex items-center gap-2">
          <h3 className="text-sm font-semibold">{t("kbview.embeddingVectorEngine")}</h3>
          <span className="ml-auto rounded-full border border-[var(--border)] bg-[var(--surface)] px-2 py-0.5 text-[11px] text-[var(--text-muted)]">{t("kbview.fixedAtCreation")}</span>
        </div>
        <div className="mt-3 grid gap-x-5 md:grid-cols-2">
          <div>
            <KV k="Embedding provider" v={str(emb.provider_id, "local_hash")} />
            <KV k={t("kbview.embeddingModelLabel")} v={str(emb.model, "local-hash-v1")} />
            <KV k={t("kbview.vectorDim")} v={num(emb.dimension, 64)} />
          </div>
          <div>
            <KV k={t("kbview.vectorEngine")} v={str(vec.provider_id, "local_sql")} />
            <KV k={t("kbview.distanceMetric")} v={str(vec.metric, "cosine")} />
          </div>
        </div>
        <p className="mt-3 text-[11px] text-[var(--text-faint)]">{t("kbview.embeddingFixedNote")}</p>
      </div>

      {/* chunking */}
      <div className={CARD}>
        <h3 className="text-sm font-semibold">{t("kbview.chunkingTitle")}</h3>
        <p className="mt-0.5 text-xs text-[var(--text-faint)]">{t("kbview.chunkingNote")}</p>
        <div className="mt-3 grid gap-3.5 md:grid-cols-2">
          <Field label={t("kbview.chunkSizeLabel")}>
            <input type="number" className={cn(INPUT, "font-mono")} value={chunkSize}
              onChange={(e) => setChunkSize(Math.max(100, Number(e.target.value) || 0))} />
          </Field>
          <Field label={t("kbview.chunkOverlapLabel")}>
            <input type="number" className={cn(INPUT, "font-mono")} value={chunkOverlap}
              onChange={(e) => setChunkOverlap(Math.max(0, Number(e.target.value) || 0))} />
          </Field>
        </div>
        <div className="mt-3 flex h-4 overflow-hidden rounded border border-[var(--border)]">
          <div className="flex-[8] bg-[var(--accent-soft)]" />
          <div className="bg-[var(--accent)]/30" style={{ flexGrow: Math.max(0.2, (chunkOverlap / Math.max(1, chunkSize)) * 8), flexShrink: 0, flexBasis: 0 }} />
          <div className="flex-[8] bg-[var(--surface-2)]" />
        </div>
        <div className="mt-1 flex justify-between text-[11px] text-[var(--text-faint)]">
          <span>{t("kbview.chunkN")}</span><span className="text-[var(--accent)]">{t("kbview.overlapN", { n: chunkOverlap })}</span><span>{t("kbview.chunkN1")}</span>
        </div>
      </div>

      {/* retrieval */}
      <div className={CARD}>
        <h3 className="text-sm font-semibold">{t("kbview.retrievalDefaults")}</h3>
        <p className="mt-0.5 text-xs text-[var(--text-faint)]">{t("kbview.retrievalDefaultsNote")}</p>
        <div className="mt-3 flex flex-wrap items-end gap-5">
          <Field label={t("kbview.mode")}>
            <Seg value={mode} onChange={setMode} options={[
              { value: "hybrid", label: "hybrid" }, { value: "vector", label: "vector" }, { value: "keyword", label: "keyword" },
            ]} />
          </Field>
          <div className="w-24"><Field label="top_k">
            <input type="number" className={cn(INPUT, "font-mono")} value={topK}
              onChange={(e) => setTopK(Math.max(1, Number(e.target.value) || 1))} />
          </Field></div>
          <div className="w-28"><Field label={t("kbview.scoreThreshold")}>
            <input type="number" step="0.05" className={cn(INPUT, "font-mono")} value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value) || 0)} />
          </Field></div>
          <div className="pb-2"><Toggle on={forceCite} onChange={setForceCite} label={t("kbview.forceCitation")} /></div>
        </div>
      </div>

      {indexAffecting && (
        <div className="flex items-start gap-2.5 rounded-[var(--radius)] border border-[var(--warning)]/40 bg-[var(--warning)]/10 px-3.5 py-3 text-[12.5px] text-[var(--warning)]">
          <TriangleAlert className="mt-0.5 h-4 w-4 flex-shrink-0" />
          <div>{t("kbview.indexAffectingA", { count: kb.chunk_count })}<b>{t("kbview.indexAffectingRebuild")}</b>{t("kbview.indexAffectingB")}</div>
        </div>
      )}

      <div className="sticky bottom-0 flex items-center gap-3 rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-3 shadow-[var(--shadow)]">
        {dirty
          ? <span className="flex items-center gap-2 text-[12.5px] text-[var(--warning)]"><span className="h-1.5 w-1.5 rounded-full bg-[var(--warning)]" /> {t("kbview.unsavedChanges")}</span>
          : <span className="text-[12.5px] text-[var(--text-faint)]">{t("kbview.up_to_date")}</span>}
        <div className="flex-1" />
        <button className={BTN_GHOST} disabled={!dirty || saving} onClick={reset}>{t("kbview.discard")}</button>
        <button className={BTN_PRIMARY} disabled={!dirty || saving} onClick={save}>
          {saving && <Loader2 className="h-4 w-4 animate-spin" />} {t("kbview.saveConfig")}
        </button>
      </div>
    </div>
  );
}

// ---------- Data sources ---------- //
function sourceSummary(ds: KnowledgeDataSource): string {
  const cfg = ds.source_config || {};
  if (ds.source_type === "yuque") {
    const base = `${cfg.group_login || "?"}/${cfg.book_slug || "?"}`;
    return typeof cfg.path === "string" && cfg.path ? `${base} · ${cfg.path}` : base;
  }
  if (typeof cfg.llms_url === "string" && cfg.llms_url) return cfg.llms_url;
  if (typeof cfg.product === "string" && cfg.product)
    return `help.aliyun.com/zh/${cfg.product}/llms.txt`;
  return "—";
}

export function SyncProgressView({ progress, cancelling, onCancel }: {
  progress: KnowledgeSyncProgress;
  cancelling: boolean;
  onCancel: () => void;
}) {
  const { t } = useI18n();
  const completed = progress.indexed + progress.unchanged + progress.failed;
  const percent = progress.total > 0
    ? Math.min(100, Math.max(0, Math.round((completed / progress.total) * 100)))
    : 0;
  const phaseKey = `kbview.syncPhase.${progress.phase}` as MessageKey;
  return (
    <div className="ml-[38px] space-y-2 rounded-[var(--radius)] bg-[var(--surface-subtle)] p-3">
      <div className="flex items-center justify-between text-[12px]">
        <span className="font-medium">{t(phaseKey)}</span>
        <span className="font-mono tabular-nums">{completed} / {progress.total}</span>
      </div>
      <div
        role="progressbar"
        aria-label={t("kbview.syncProgress")}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        className="h-2 overflow-hidden rounded-full bg-[var(--border)]"
      >
        <div className="h-full rounded-full bg-[var(--accent)] transition-[width]" style={{ width: `${percent}%` }} />
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-[var(--text-muted)]">
        <span>{t("kbview.syncThroughput", { rate: progress.docs_per_second ?? 0 })}</span>
        {progress.estimated_seconds_remaining != null && (
          <span>{t("kbview.syncEta", { seconds: progress.estimated_seconds_remaining })}</span>
        )}
        {progress.failed > 0 && <span className="text-[var(--danger)]">{t("kbview.syncProgressFailed", { count: progress.failed })}</span>}
        <button className={cn(BTN_GHOST, "ml-auto py-1 text-[11px]")} disabled={cancelling} onClick={onCancel}>
          {cancelling ? t("kbview.cancelling") : t("kbview.cancelSync")}
        </button>
      </div>
    </div>
  );
}

function DataSourcePanel({ kb, onChanged }: { kb: KnowledgeBase; onChanged: () => Promise<void> }) {
  const { t } = useI18n();
  const [sources, setSources] = useState<KnowledgeDataSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState<KnowledgeDataSource | "new" | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [cancellingId, setCancellingId] = useState<string | null>(null);

  const refresh = async () => {
    try { setSources(await listDataSources(kb.id)); }
    catch (e) { toast.error(e instanceof Error ? e.message : t("kbview.loadDsFailed")); }
    finally { setLoading(false); }
  };
  useEffect(() => { setLoading(true); void refresh(); }, [kb.id]);

  // active_job_id is authoritative; status can be stale during lease recovery.
  useEffect(() => {
    if (!sources.some((s) => !!s.active_job_id || s.status === "syncing")) return;
    const timer = setTimeout(async () => { await refresh(); void onChanged(); }, 2500);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sources]);

  const runSync = async (ds: KnowledgeDataSource) => {
    setBusyId(ds.id);
    try {
      await syncDataSource(kb.id, ds.id);
      toast.success(t("kbview.syncStarted"));
      await refresh();
    } catch (e) { toast.error(e instanceof Error ? e.message : t("kbview.syncFailedToast")); }
    finally { setBusyId(null); }
  };

  const cancelSync = async (ds: KnowledgeDataSource) => {
    setCancellingId(ds.id);
    try {
      await cancelDataSourceSync(kb.id, ds.id);
      await refresh();
    } catch (e) { toast.error(e instanceof Error ? e.message : t("kbview.cancelSyncFailed")); }
  };

  const toggleEnabled = async (ds: KnowledgeDataSource) => {
    try { await updateDataSource(kb.id, ds.id, { enabled: !ds.enabled }); await refresh(); }
    catch (e) { toast.error(e instanceof Error ? e.message : t("kbview.updateFailed")); }
  };

  const remove = async (ds: KnowledgeDataSource) => {
    if (!confirm(t("kbview.confirmDeleteDs", { name: ds.name }))) return;
    try { await deleteDataSource(kb.id, ds.id); toast.success(t("kbview.deleted")); await refresh(); }
    catch (e) { toast.error(e instanceof Error ? e.message : t("kbview.deleteFailed")); }
  };

  return (
    <div className="space-y-3.5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <p className="max-w-2xl text-[13px] text-[var(--text-muted)]">
          {t("kbview.dsIntroA")}<b className="text-[var(--text)]">{t("kbview.dsIntroAliyun")}</b>{t("kbview.dsIntroB")}
          <code className="font-mono text-[11px]">llms.txt</code>{t("kbview.dsIntroManifest")}
        </p>
        <button className={BTN_PRIMARY} onClick={() => setEditing("new")}>
          <Plus className="h-4 w-4" /> {t("kbview.addDataSource")}
        </button>
      </div>

      {loading ? (
        <div className="flex items-center gap-2 py-10 text-sm text-[var(--text-muted)]">
          <Loader2 className="h-4 w-4 animate-spin" /> {t("common.loading")}
        </div>
      ) : sources.length === 0 ? (
        <div className="grid place-items-center rounded-[var(--radius-lg)] border border-dashed border-[var(--border)] py-16 text-center">
          <Globe className="h-6 w-6 text-[var(--text-faint)]" />
          <div className="mt-3 text-sm text-[var(--text-muted)]">{t("kbview.noDataSources")}</div>
          <button className={cn(BTN_GHOST, "mt-3")} onClick={() => setEditing("new")}>
            <Plus className="h-4 w-4" /> {t("kbview.addAliyunDs")}
          </button>
        </div>
      ) : (
        <div className="grid gap-3">
          {sources.map((ds) => {
            const syncing = ds.status === "syncing" || !!ds.active_job_id;
            const rep = ds.last_sync_report || {};
            return (
              <div key={ds.id} className={cn(CARD, "flex flex-col gap-2.5")}>
                <div className="flex items-start gap-2.5">
                  <span className="mt-0.5 grid h-7 w-7 flex-shrink-0 place-items-center rounded-[var(--radius-sm)] bg-[var(--accent-soft)] text-[var(--accent)]">
                    <Globe className="h-4 w-4" />
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[14px] font-semibold">{ds.name}</span>
                      <Pill status={ds.source_type} />
                      {!ds.enabled && <span className="text-[11px] text-[var(--text-faint)]">{t("kbview.dsDisabled")}</span>}
                    </div>
                    <div className="mt-0.5 truncate font-mono text-[11px] text-[var(--text-faint)]">{sourceSummary(ds)}</div>
                  </div>
                  <Pill status={ds.status} />
                </div>

                {ds.active_job_id && ds.sync_progress && (
                  <SyncProgressView
                    progress={ds.sync_progress}
                    cancelling={cancellingId === ds.id}
                    onCancel={() => void cancelSync(ds)}
                  />
                )}

                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 pl-[38px] text-[12px] text-[var(--text-muted)]">
                  <span><b className="font-semibold text-[var(--text)] tabular-nums">{ds.doc_count}</b> {t("kbview.docs")}</span>
                  <span>{t("kbview.lastSync", { time: fmtTime(t, ds.last_sync_finished_at || ds.last_sync_at || undefined) })}</span>
                  {(ds.status === "succeeded" || ds.status === "partial") && (
                    <span className="font-mono text-[11px] text-[var(--text-faint)]">
                      {t("kbview.syncReport", { added: rep.added ?? 0, updated: rep.updated ?? 0, deleted: rep.deleted ?? 0 })}
                      {typeof rep.failed === "number" && rep.failed > 0 ? t("kbview.syncFailedCount", { failed: rep.failed }) : ""}
                    </span>
                  )}
                </div>

                {ds.last_error && (
                  <div className="ml-[38px] flex items-start gap-1.5 rounded-[var(--radius)] border border-[var(--danger)]/30 bg-[var(--danger)]/10 px-2.5 py-1.5 text-[12px] text-[var(--danger)]">
                    <TriangleAlert className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
                    <span className="break-all">{ds.last_error}</span>
                  </div>
                )}

                <div className="flex items-center gap-2 border-t border-[var(--border)] pt-2.5">
                  <Toggle on={ds.enabled} onChange={() => void toggleEnabled(ds)} label={ds.enabled ? t("kbview.dsEnabled") : t("kbview.dsDisabled")} />
                  <div className="flex-1" />
                  <button
                    className={BTN_GHOST} disabled={syncing || busyId === ds.id || !ds.enabled}
                    onClick={() => void runSync(ds)} title={ds.enabled ? t("kbview.syncNow") : t("kbview.enableToSync")}
                  >
                    <RefreshCw className={cn("h-4 w-4", (syncing || busyId === ds.id) && "animate-spin")} />
                    {syncing ? t("kbview.status.syncing") : t("kbview.sync")}
                  </button>
                  <button className={ICON_BTN} title={t("common.edit")} onClick={() => setEditing(ds)}><Pencil className="h-4 w-4" /></button>
                  <button className={cn(ICON_BTN, "hover:text-[var(--danger)]")} title={t("common.delete")} onClick={() => void remove(ds)}><Trash2 className="h-4 w-4" /></button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <DataSourceDrawer
        key={editing === "new" ? "new" : editing?.id ?? "closed"}
        kb={kb} editing={editing}
        onClose={() => setEditing(null)}
        onDone={async () => { setEditing(null); await refresh(); }}
      />
    </div>
  );
}

function DataSourceDrawer({ kb, editing, onClose, onDone }: {
  kb: KnowledgeBase; editing: KnowledgeDataSource | "new" | null;
  onClose: () => void; onDone: () => Promise<void>;
}) {
  const { t } = useI18n();
  const open = editing !== null;
  const isEdit = editing !== null && editing !== "new";
  const src = isEdit ? editing.source_config || {} : {};
  const initUrl = typeof src.llms_url === "string" ? src.llms_url : "";
  const strv = (v: unknown, d = "") => (typeof v === "string" ? v : d);

  // Source type is chosen on create and fixed on edit (changing it would
  // orphan the ingested docs). Existing rows infer it from source_type.
  const initType: "llms_txt" | "yuque" = isEdit && editing.source_type === "yuque" ? "yuque" : "llms_txt";
  const [type, setType] = useState<"llms_txt" | "yuque">(initType);
  const [name, setName] = useState(isEdit ? editing.name : "");
  const [busy, setBusy] = useState(false);

  // llms_txt fields
  const [mode, setMode] = useState<"product" | "url">(initUrl ? "url" : "product");
  const [product, setProduct] = useState(strv(src.product));
  const [llmsUrl, setLlmsUrl] = useState(initUrl);
  const [sections, setSections] = useState(Array.isArray(src.sections) ? (src.sections as string[]).join(", ") : "");
  const [lang, setLang] = useState(strv(src.lang, "zh"));

  // yuque fields
  const [group, setGroup] = useState(strv(src.group_login));
  const [book, setBook] = useState(strv(src.book_slug));
  const [tokenEnv, setTokenEnv] = useState(strv(src.token_env, "YUQUE_TOKEN"));
  const [yqPath, setYqPath] = useState(strv(src.path));
  const [apiBase, setApiBase] = useState(strv(src.api_base));
  const [webBase, setWebBase] = useState(strv(src.web_base));

  const valid = name.trim() !== "" && (
    type === "yuque"
      ? group.trim() !== "" && book.trim() !== "" && tokenEnv.trim() !== ""
      : mode === "product" ? product.trim() !== "" : llmsUrl.trim() !== ""
  );

  const buildConfig = (): Record<string, unknown> => {
    if (type === "yuque") {
      const cfg: Record<string, unknown> = {
        group_login: group.trim(), book_slug: book.trim(), token_env: tokenEnv.trim(),
      };
      if (yqPath.trim()) cfg.path = yqPath.trim();
      if (apiBase.trim()) cfg.api_base = apiBase.trim();
      if (webBase.trim()) cfg.web_base = webBase.trim();
      return cfg;
    }
    const cfg: Record<string, unknown> = {};
    if (mode === "product") cfg.product = product.trim();
    else cfg.llms_url = llmsUrl.trim();
    const secs = sections.split(",").map((s) => s.trim()).filter(Boolean);
    if (secs.length) cfg.sections = secs;
    cfg.lang = lang.trim() || "zh";
    return cfg;
  };

  const submit = async () => {
    if (!valid) return;
    setBusy(true);
    try {
      const cfg = buildConfig();
      if (isEdit) {
        await updateDataSource(kb.id, editing.id, { name: name.trim(), source_config: cfg });
        toast.success(t("kbview.dsUpdated"));
      } else {
        await createDataSource(kb.id, { name: name.trim(), source_type: type, source_config: cfg });
        toast.success(t("kbview.dsAdded"));
      }
      await onDone();
    } catch (e) { toast.error(e instanceof Error ? e.message : t("kbview.saveFailed")); }
    finally { setBusy(false); }
  };

  return (
    <Drawer open={open} title={isEdit ? t("kbview.editDataSource") : t("kbview.addDataSource")} onClose={onClose}
      footer={<>
        <button className={BTN_GHOST} onClick={onClose}>{t("common.cancel")}</button>
        <button className={BTN_PRIMARY} disabled={busy || !valid} onClick={submit}>
          {busy && <Loader2 className="h-4 w-4 animate-spin" />} {isEdit ? t("common.save") : t("common.add")}
        </button>
      </>}>
      <Field label={t("kbview.type")}>
        {isEdit ? (
          <div className="flex items-center gap-2 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-[13px]">
            {type === "yuque" ? <BookOpen className="h-4 w-4 text-[var(--accent)]" /> : <Globe className="h-4 w-4 text-[var(--accent)]" />}
            {type === "yuque" ? t("kbview.yuqueKb") : t("kbview.aliyunHelpDocs")}
          </div>
        ) : (
          <Seg value={type} onChange={setType} options={[
            { value: "llms_txt", label: t("kbview.status.llms_txt") }, { value: "yuque", label: t("kbview.status.yuque") },
          ]} />
        )}
      </Field>
      <Field label={t("kbview.name")}><input className={INPUT} value={name} onChange={(e) => setName(e.target.value)} placeholder={type === "yuque" ? t("kbview.dsNamePlaceholderYuque") : t("kbview.dsNamePlaceholderLlms")} /></Field>

      {type === "yuque" ? (<>
        <Field label={t("kbview.groupLogin")} hint={t("kbview.groupLoginHint")}>
          <input className={cn(INPUT, "font-mono text-xs")} value={group} onChange={(e) => setGroup(e.target.value)} placeholder="acme" />
        </Field>
        <Field label={t("kbview.bookSlug")} hint={t("kbview.bookSlugHint")}>
          <input className={cn(INPUT, "font-mono text-xs")} value={book} onChange={(e) => setBook(e.target.value)} placeholder="handbook" />
        </Field>
        <Field label={t("kbview.tokenEnv")} hint={t("kbview.tokenEnvHint")}>
          <input className={cn(INPUT, "font-mono text-xs")} value={tokenEnv} onChange={(e) => setTokenEnv(e.target.value)} placeholder="YUQUE_TOKEN" />
        </Field>
        <Field label={t("kbview.yqPath")} hint={t("kbview.yqPathHint")}>
          <input className={INPUT} value={yqPath} onChange={(e) => setYqPath(e.target.value)} placeholder={t("kbview.yqPathPlaceholder")} />
        </Field>
        <Field label={t("kbview.apiBase")} hint={t("kbview.apiBaseHint")}>
          <input className={cn(INPUT, "font-mono text-xs")} value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="https://www.yuque.com/api/v2" />
        </Field>
        <Field label={t("kbview.webBase")} hint={t("kbview.webBaseHint")}>
          <input className={cn(INPUT, "font-mono text-xs")} value={webBase} onChange={(e) => setWebBase(e.target.value)} placeholder="https://www.yuque.com" />
        </Field>
        <div className="flex items-start gap-2 rounded-[var(--radius)] border border-[var(--accent)]/30 bg-[var(--accent-soft)] px-3 py-2.5 text-[12px] text-[var(--accent)]">
          <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
          <div>{t("kbview.yuqueInfoA")}<code className="font-mono">{tokenEnv.trim() || "YUQUE_TOKEN"}</code>{t("kbview.yuqueInfoB")}<code className="font-mono">PAIRAG_DATASOURCE_ALLOW_PRIVATE_NETWORK</code>{t("kbview.yuqueInfoC")}</div>
        </div>
      </>) : (<>
        <Field label={t("kbview.sourceMethod")}>
          <Seg value={mode} onChange={setMode} options={[
            { value: "product", label: t("kbview.productId") }, { value: "url", label: t("kbview.fullUrl") },
          ]} />
        </Field>
        {mode === "product" ? (
          <Field label={t("kbview.productField")} hint={t("kbview.productHint")}>
            <input className={cn(INPUT, "font-mono text-xs")} value={product}
              onChange={(e) => setProduct(e.target.value)} placeholder="pai" />
          </Field>
        ) : (
          <Field label="llms.txt URL" hint={t("kbview.llmsUrlHint")}>
            <input className={cn(INPUT, "font-mono text-xs")} value={llmsUrl}
              onChange={(e) => setLlmsUrl(e.target.value)} placeholder="https://help.aliyun.com/zh/pai/llms.txt" />
          </Field>
        )}
        <Field label={t("kbview.sections")} hint={t("kbview.sectionsHint")}>
          <input className={INPUT} value={sections} onChange={(e) => setSections(e.target.value)} placeholder={t("kbview.sectionsPlaceholder")} />
        </Field>
        <Field label={t("kbview.langField")} hint={t("kbview.langHint")}>
          <input className={cn(INPUT, "w-24")} value={lang} onChange={(e) => setLang(e.target.value)} placeholder="zh" />
        </Field>
        <div className="flex items-start gap-2 rounded-[var(--radius)] border border-[var(--accent)]/30 bg-[var(--accent-soft)] px-3 py-2.5 text-[12px] text-[var(--accent)]">
          <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
          <div>{t("kbview.llmsInfoA")}<code className="font-mono">.md</code>{t("kbview.llmsInfoB")}</div>
        </div>
      </>)}
    </Drawer>
  );
}

// ---------- Files ---------- //
function FilesPanel({ kb, onChanged }: { kb: KnowledgeBase; onChanged: () => Promise<void> }) {
  const { t } = useI18n();
  const PAGE = 50;
  const [docs, setDocs] = useState<KnowledgeDocument[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [q, setQ] = useState("");
  const [showImport, setShowImport] = useState(false);
  const [viewDoc, setViewDoc] = useState<KnowledgeDocument | null>(null);

  const load = async (opts?: { append?: boolean; query?: string }) => {
    const append = opts?.append ?? false;
    const query = opts?.query ?? q;
    if (append) setLoadingMore(true); else setLoading(true);
    try {
      const page = await listKnowledgeDocuments(kb.id, {
        query: query.trim() || undefined,
        limit: PAGE,
        offset: append ? docs.length : 0,
      });
      setTotal(page.total);
      setDocs((prev) => (append ? [...prev, ...page.data] : page.data));
    } catch {
      if (!append) { setDocs([]); setTotal(0); }
    } finally {
      if (append) setLoadingMore(false); else setLoading(false);
    }
  };

  // Reset + reload on KB switch or (debounced) query change; server-side filter.
  useEffect(() => {
    const timer = setTimeout(() => { void load({ append: false, query: q }); }, q ? 250 : 0);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kb.id, q]);

  const refresh = () => load({ append: false });
  const hasMore = docs.length < total;

  // poll while any doc is still processing (queued/running ingest job); refresh
  // the KB counts once it settles to indexed. Mirrors the datasource poller.
  useEffect(() => {
    if (!docs.some((d) => d.status === "processing")) return;
    const timer = setTimeout(async () => { await refresh(); void onChanged(); }, 2500);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docs]);

  return (
    <div className="space-y-3.5">
      <div className="flex flex-wrap items-center gap-2.5">
        <div className="relative max-w-[320px] flex-1">
          <Search className="absolute top-1/2 left-2.5 h-3.5 w-3.5 -translate-y-1/2 text-[var(--text-faint)]" />
          <input className={cn(INPUT, "pl-8")} placeholder={t("kbview.searchDocs")} value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <div className="flex-1" />
        {!loading && <span className="text-[12px] text-[var(--text-faint)] tabular-nums">{t("kbview.totalDocs", { total })}</span>}
        <button className={BTN_PRIMARY} onClick={() => setShowImport(true)}><Plus className="h-4 w-4" /> {t("kbview.importDoc")}</button>
      </div>

      <div className={cn(CARD, "p-1")}>
        {loading ? (
          <div className="flex items-center gap-2 p-6 text-sm text-[var(--text-muted)]"><Loader2 className="h-4 w-4 animate-spin" /> {t("common.loading")}</div>
        ) : docs.length === 0 ? (
          <div className="p-8 text-center text-sm text-[var(--text-muted)]">{q ? t("kbview.noMatchDocs") : t("kbview.noDocs")}</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-[13px]">
              <thead>
                <tr className="text-left text-[11px] font-semibold tracking-wide text-[var(--text-faint)] uppercase">
                  <th className="border-b border-[var(--border)] px-3 py-2">{t("kbview.colTitle")}</th>
                  <th className="border-b border-[var(--border)] px-3 py-2">{t("kbview.colSource")}</th>
                  <th className="border-b border-[var(--border)] px-3 py-2">{t("kbview.statusLabel")}</th>
                  <th className="border-b border-[var(--border)] px-3 py-2 text-right">{t("kbview.colChunks")}</th>
                  <th className="border-b border-[var(--border)] px-3 py-2">{t("kbview.colTags")}</th>
                  <th className="border-b border-[var(--border)] px-3 py-2 text-right">{t("kbview.colIndexedAt")}</th>
                  <th className="border-b border-[var(--border)] px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {docs.map((d) => (
                  <tr key={d.id} className="group hover:bg-[var(--surface)]">
                    <td className="border-b border-[var(--border)] px-3 py-2.5">
                      <div className="font-medium">{d.title}</div>
                      <div className="max-w-[280px] truncate font-mono text-[11px] text-[var(--text-faint)]">{d.uri}</div>
                    </td>
                    <td className="border-b border-[var(--border)] px-3 py-2.5"><Pill status={d.source_type} /></td>
                    <td className="border-b border-[var(--border)] px-3 py-2.5"><Pill status={d.status} /></td>
                    <td className="border-b border-[var(--border)] px-3 py-2.5 text-right tabular-nums">{d.chunk_count}</td>
                    <td className="border-b border-[var(--border)] px-3 py-2.5">
                      <div className="flex flex-wrap gap-1">
                        {(d.tags || []).slice(0, 3).map((tg) => (
                          <span key={tg} className="rounded-full bg-[var(--surface-2)] px-1.5 py-0.5 text-[10.5px] text-[var(--text-muted)]">{tg}</span>
                        ))}
                      </div>
                    </td>
                    <td className="border-b border-[var(--border)] px-3 py-2.5 text-right text-[var(--text-faint)]">{fmtTime(t, d.indexed_at)}</td>
                    <td className="border-b border-[var(--border)] px-3 py-2.5">
                      <div className="flex justify-end gap-1 opacity-60 group-hover:opacity-100">
                        <button className={ICON_BTN} title={t("kbview.viewChunks")} onClick={() => setViewDoc(d)}><Eye className="h-4 w-4" /></button>
                        <button className={ICON_BTN} title={t("kbview.reindexSoon")} disabled><RefreshCw className="h-4 w-4" /></button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {!loading && hasMore && (
          <div className="flex justify-center p-2">
            <button className={BTN_GHOST} disabled={loadingMore} onClick={() => void load({ append: true })}>
              {loadingMore ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              {t("kbview.loadMore", { loaded: docs.length, total })}
            </button>
          </div>
        )}
      </div>

      <ImportDrawer
        kb={kb} open={showImport} onClose={() => setShowImport(false)}
        onDone={async () => { setShowImport(false); await refresh(); await onChanged(); }}
      />
      <ChunkDrawer kb={kb} doc={viewDoc} onClose={() => setViewDoc(null)} />
    </div>
  );
}

function ChunkDrawer({ kb, doc, onClose }: { kb: KnowledgeBase; doc: KnowledgeDocument | null; onClose: () => void }) {
  const { t } = useI18n();
  const PAGE = 50;
  const [chunks, setChunks] = useState<KnowledgeChunk[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);

  const load = async (append: boolean) => {
    if (!doc) return;
    if (append) setLoadingMore(true); else setLoading(true);
    try {
      const page = await listKnowledgeChunks(kb.id, doc.id, { limit: PAGE, offset: append ? chunks.length : 0 });
      setTotal(page.total);
      setChunks((prev) => (append ? [...prev, ...page.data] : page.data));
    } catch {
      if (!append) { setChunks([]); setTotal(0); }
    } finally {
      if (append) setLoadingMore(false); else setLoading(false);
    }
  };

  useEffect(() => {
    if (!doc) return;
    void load(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kb.id, doc?.id]);

  const hasMore = chunks.length < total;
  return (
    <Drawer open={!!doc} title={doc ? t("kbview.chunksTitle", { title: doc.title, total }) : t("kbview.chunksTitleFallback")} onClose={onClose}
      footer={<button className={BTN_GHOST} onClick={onClose}>{t("common.close")}</button>}>
      {loading ? (
        <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]"><Loader2 className="h-4 w-4 animate-spin" /> {t("common.loading")}</div>
      ) : chunks.length === 0 ? (
        <div className="text-sm text-[var(--text-muted)]">{t("kbview.noChunks")}</div>
      ) : (
        <div className="space-y-2">
          {chunks.map((c) => (
            <div key={c.id} className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-3">
              <div className="mb-1 flex justify-between text-[11px] text-[var(--text-muted)]">
                <span>#{c.chunk_index} · {c.token_count} tokens</span><Pill status={c.status} />
              </div>
              <ChunkBody text={c.text} headingPath={c.heading_path} charStart={c.char_start} charEnd={c.char_end} />
            </div>
          ))}
          {hasMore && (
            <div className="flex justify-center pt-1">
              <button className={BTN_GHOST} disabled={loadingMore} onClick={() => void load(true)}>
                {loadingMore ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                {t("kbview.loadMore", { loaded: chunks.length, total })}
              </button>
            </div>
          )}
        </div>
      )}
    </Drawer>
  );
}

function ImportDrawer({ kb, open, onClose, onDone }: {
  kb: KnowledgeBase; open: boolean; onClose: () => void; onDone: () => Promise<void>;
}) {
  const { t } = useI18n();
  const [form, setForm] = useState({ title: "", uri: "", tags: "", content: "" });
  const [busy, setBusy] = useState(false);
  const [accept, setAccept] = useState<string>("");
  const [maxMb, setMaxMb] = useState<number>(20);
  useEffect(() => {
    if (!open) return;
    getUploadSupport()
      .then((s) => { setAccept(s.extensions.join(",")); setMaxMb(s.max_mb); })
      .catch(() => {});
  }, [open]);
  const submit = async () => {
    if (!form.title.trim() || !form.content.trim()) return;
    setBusy(true);
    try {
      await importKnowledgeDocument(kb.id, {
        title: form.title.trim(),
        uri: form.uri.trim() || undefined,
        source_type: form.uri.trim().startsWith("http") ? "website" : "text",
        content: form.content,
        tags: form.tags.split(",").map((s) => s.trim()).filter(Boolean),
      });
      setForm({ title: "", uri: "", tags: "", content: "" });
      toast.success(t("kbview.imported"));
      await onDone();
    } catch (e) { toast.error(e instanceof Error ? e.message : t("kbview.importFailed")); }
    finally { setBusy(false); }
  };
  const upload = async (file: File) => {
    if (file.size > maxMb * 1024 * 1024) {
      toast.error(t("kbview.fileTooLarge", { mb: maxMb }));
      return;
    }
    setBusy(true);
    try {
      await uploadKnowledgeDocument(kb.id, file, {
        title: form.title.trim() || undefined,
        tags: form.tags.split(",").map((s) => s.trim()).filter(Boolean),
      });
      toast.success(t("kbview.uploaded"));
      await onDone();
    } catch (e) { toast.error(e instanceof Error ? e.message : t("kbview.uploadFailed")); }
    finally { setBusy(false); }
  };
  return (
    <Drawer open={open} title={t("kbview.importDoc")} onClose={onClose}
      footer={<>
        <button className={BTN_GHOST} onClick={onClose}>{t("common.cancel")}</button>
        <button className={BTN_PRIMARY} disabled={busy || !form.title.trim() || !form.content.trim()} onClick={submit}>
          {busy && <Loader2 className="h-4 w-4 animate-spin" />} {t("kbview.importAndIndex")}
        </button>
      </>}>
      <Field label={t("kbview.uploadFile")} hint={accept ? t("kbview.uploadHintExt", { accept, mb: maxMb }) : t("kbview.uploadHintDefault")}>
        <label className={cn(BTN_GHOST, "w-full cursor-pointer")}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
          {t("kbview.chooseFile")}
          <input type="file" accept={accept || undefined} className="hidden" disabled={busy}
            onChange={(e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (f) void upload(f);
            }} />
        </label>
      </Field>
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-[var(--text-muted)]">
        <div className="h-px flex-1 bg-[var(--border)]" />{t("kbview.orPaste")}<div className="h-px flex-1 bg-[var(--border)]" />
      </div>
      <Field label={t("kbview.colTitle")}><input className={INPUT} value={form.title} onChange={(e) => setForm((s) => ({ ...s, title: e.target.value }))} placeholder={t("kbview.docTitlePlaceholder")} /></Field>
      <Field label={t("kbview.sourceUri")} hint={t("kbview.sourceUriHint")}>
        <input className={cn(INPUT, "font-mono text-xs")} value={form.uri} onChange={(e) => setForm((s) => ({ ...s, uri: e.target.value }))} placeholder="https://…" />
      </Field>
      <Field label={t("kbview.tagsComma")}><input className={INPUT} value={form.tags} onChange={(e) => setForm((s) => ({ ...s, tags: e.target.value }))} placeholder="eas, product-docs" /></Field>
      <Field label={t("kbview.content")}>
        <textarea rows={9} className={cn(INPUT, "resize-none")} value={form.content}
          onChange={(e) => setForm((s) => ({ ...s, content: e.target.value }))}
          placeholder={t("kbview.contentPlaceholder")} />
      </Field>
      <div className="flex items-start gap-2 rounded-[var(--radius)] border border-[var(--accent)]/30 bg-[var(--accent-soft)] px-3 py-2.5 text-[12px] text-[var(--accent)]">
        <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
        <div>{t("kbview.importInfo")}</div>
      </div>
    </Drawer>
  );
}

// ---------- Recall ---------- //
// Ties the recall surface to the agent goal: tells the tester that a chat agent
// can retrieve from this KB via the knowledge_search tool, and hands over the id
// for scoping a call. Visibility gates whether other users' agents can reach it.
function AgentAvailability({ kb, queryable }: { kb: KnowledgeBase; queryable: boolean }) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    if (await copyText(kb.id)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } else {
      toast.error(t("common.copyFailed"));
    }
  };
  return (
    <div className={cn(CARD, "flex flex-wrap items-center gap-x-4 gap-y-2.5")}>
      <span className="grid h-8 w-8 flex-shrink-0 place-items-center rounded-[var(--radius-sm)] bg-[var(--accent-soft)] text-[var(--accent)]">
        <Bot className="h-4 w-4" />
      </span>
      <div className="min-w-0 flex-1">
        <div className="text-[13px] font-semibold">{t("kbview.agentConnected")}</div>
        <div className="mt-0.5 text-[12px] text-[var(--text-muted)]">
          {t("kbview.agentAvailA")}<code className="font-mono text-[11px]">knowledge_search</code>{t("kbview.agentAvailB")}
          {queryable ? t("kbview.agentAvailQueryable") : t("kbview.agentAvailPrivate")}
        </div>
      </div>
      <button
        className={cn(BTN_GHOST, "font-mono text-[11px]")} onClick={copy}
        title={t("kbview.copyKbIdTitle")}
      >
        {copied ? <Check className="h-3.5 w-3.5 text-[var(--success)]" /> : <Copy className="h-3.5 w-3.5" />}
        {copied ? t("common.copied") : `id: ${kb.id.slice(0, 8)}…`}
      </button>
    </div>
  );
}

function RecallPanel({ kb }: { kb: KnowledgeBase }) {
  const { t } = useI18n();
  const init = retrievalOf(kb);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState(init.mode);
  const [topK, setTopK] = useState(init.top_k);
  const [threshold, setThreshold] = useState(init.score_threshold);
  const [tag, setTag] = useState("");
  const [hits, setHits] = useState<KnowledgeHit[] | null>(null);
  const [total, setTotal] = useState(0);
  const [busy, setBusy] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [lastQuery, setLastQuery] = useState("");
  const { engine } = useEngineStatus();

  const fetchPage = async (offset: number) => {
    const filters = tag.trim() ? { tags: [tag.trim()] } : {};
    return searchKnowledge({ kb_ids: [kb.id], query, mode, top_k: topK, offset, score_threshold: threshold, filters });
  };

  const run = async () => {
    if (!query.trim()) return;
    setBusy(true);
    try {
      const page = await fetchPage(0);
      setHits(page.data);
      setTotal(page.total);
      setLastQuery(query);
    } catch (e) { toast.error(e instanceof Error ? e.message : t("kbview.searchFailed")); }
    finally { setBusy(false); }
  };

  const loadMore = async () => {
    if (!hits) return;
    setLoadingMore(true);
    try {
      const page = await fetchPage(hits.length);
      setHits((prev) => [...(prev || []), ...page.data]);
      setTotal(page.total);
    } catch (e) { toast.error(e instanceof Error ? e.message : t("kbview.loadMoreFailed")); }
    finally { setLoadingMore(false); }
  };

  const queryable = kb.visibility === "workspace" || kb.visibility === "public";
  const hasMore = hits !== null && hits.length < total;

  return (
    <div className="space-y-3.5">
      <AgentAvailability kb={kb} queryable={queryable} />
      <div className="flex items-center gap-2">
        <span className="text-[12px] text-[var(--text-faint)]">{t("kbview.retrievalEngine")}</span>
        <EngineStatusBadge engine={engine} />
      </div>
      <div className={CARD}>
        <div className="flex gap-2">
          <input
            className={cn(INPUT, "flex-1")} value={query} placeholder={t("kbview.recallPlaceholder")}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") void run(); }}
          />
          <button className={BTN_PRIMARY} disabled={busy || !query.trim()} onClick={run}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />} {t("kbview.search")}
          </button>
        </div>
        <div className="mt-3.5 flex flex-wrap items-center gap-x-5 gap-y-3 border-t border-[var(--border)] pt-3.5 text-[12.5px] text-[var(--text-muted)]">
          <span className="flex items-center gap-2">{t("kbview.mode")}
            <Seg value={mode} onChange={setMode} options={[
              { value: "hybrid", label: "hybrid" }, { value: "vector", label: "vector" }, { value: "keyword", label: "keyword" },
            ]} />
          </span>
          <span className="flex items-center gap-2">top_k
            <input type="number" className={cn(INPUT, "w-16 px-2 py-1 font-mono")} value={topK} onChange={(e) => setTopK(Math.max(1, Number(e.target.value) || 1))} />
          </span>
          <span className="flex items-center gap-2">{t("kbview.threshold")}
            <input type="number" step="0.05" className={cn(INPUT, "w-20 px-2 py-1 font-mono")} value={threshold} onChange={(e) => setThreshold(Number(e.target.value) || 0)} />
          </span>
          <span className="flex items-center gap-2">{t("kbview.tagLabel")}
            <input className={cn(INPUT, "w-28 px-2 py-1")} value={tag} onChange={(e) => setTag(e.target.value)} placeholder={t("common.optional")} />
          </span>
        </div>
      </div>

      {hits === null ? (
        <div className="grid place-items-center rounded-[var(--radius-lg)] border border-dashed border-[var(--border)] py-14 text-center">
          <Search className="h-6 w-6 text-[var(--text-faint)]" />
          <div className="mt-3 text-sm text-[var(--text-muted)]">{t("kbview.recallEmptyTitle")}</div>
          <div className="mt-1 text-[12px] text-[var(--text-faint)]">{t("kbview.recallEmptyHintA")}<code className="font-mono">knowledge_search</code>{t("kbview.recallEmptyHintB")}</div>
        </div>
      ) : hits.length === 0 ? (
        <div className="grid place-items-center rounded-[var(--radius-lg)] border border-dashed border-[var(--border)] py-14 text-center">
          <Search className="h-6 w-6 text-[var(--text-faint)]" />
          <div className="mt-3 text-sm text-[var(--text-muted)]">{t("kbview.noHits")}</div>
          <div className="mt-1 text-[12px] text-[var(--text-faint)]">{t("kbview.noHitsHint")}</div>
        </div>
      ) : (
        <>
          <div className="text-[12.5px] text-[var(--text-muted)]">
            {t("kbview.hitCountA")}<b className="text-[var(--text)]">{total}</b>{t("kbview.hitCountB", { shown: hits.length, mode })}
          </div>
          {hits.map((h, i) => <Hit key={h.chunk_id} hit={h} rank={i + 1} query={lastQuery} />)}
          {hasMore && (
            <div className="flex justify-center pt-1">
              <button className={BTN_GHOST} disabled={loadingMore} onClick={() => void loadMore()}>
                {loadingMore ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                {t("kbview.loadMore", { loaded: hits.length, total })}
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Hit({ hit, rank, query }: { hit: KnowledgeHit; rank: number; query: string }) {
  const { t } = useI18n();
  const meta = hit.metadata ?? {};
  const chunkIdx = meta.chunk_index;
  const headingPath = Array.isArray(meta.heading_path) ? (meta.heading_path as string[]) : undefined;
  const charStart = typeof meta.char_start === "number" ? meta.char_start : undefined;
  const charEnd = typeof meta.char_end === "number" ? meta.char_end : undefined;
  return (
    <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-elevated)] p-3.5">
      <div className="flex items-center gap-2.5">
        <span className="w-6 font-mono text-[11px] text-[var(--text-faint)]">#{rank}</span>
        <span className="text-[13px] font-semibold">{hit.title}</span>
        <span className="truncate font-mono text-[11px] text-[var(--text-faint)]">
          {hit.source_uri}{typeof chunkIdx === "number" ? ` · chunk ${chunkIdx}` : ""}
        </span>
        <span className="ml-auto flex items-center gap-1.5 text-xs text-[var(--text-muted)]">
          {t("kbview.finalScore")}
          <b className="font-mono font-semibold text-[var(--accent)]">{hit.score.toFixed(3)}</b>
        </span>
      </div>
      <ChunkBody className="mt-2" text={hit.text} query={query} headingPath={headingPath} charStart={charStart} charEnd={charEnd} />
    </div>
  );
}

// ---------- Create ---------- //
function CreateDrawer({ open, onClose, onCreated }: {
  open: boolean; onClose: () => void; onCreated: (id: string) => void;
}) {
  const { t } = useI18n();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [visibility, setVisibility] = useState("private");
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(150);
  const [embeddingModel, setEmbeddingModel] = useState("");  // "" → catalog default
  const [busy, setBusy] = useState(false);
  const cat = useModelCatalog(open);
  const hasEmbedding = cat.embedding.length > 0;

  const submit = async () => {
    if (!name.trim()) return;
    setBusy(true);
    try {
      const kb = await createKnowledgeBase({
        name: name.trim(), description: description.trim(), visibility,
        default_parser_config: { chunk_size: chunkSize, chunk_overlap: chunkOverlap },
        ...(embeddingModel ? { embedding_model: embeddingModel } : {}),
      });
      setName(""); setDescription(""); setVisibility("private");
      setChunkSize(1000); setChunkOverlap(150);
      setEmbeddingModel("");
      toast.success(t("kbview.kbCreated"));
      onCreated(kb.id);
    } catch (e) { toast.error(e instanceof Error ? e.message : t("kbview.createFailed")); }
    finally { setBusy(false); }
  };

  return (
    <Drawer open={open} title={t("kbview.newKb")} onClose={onClose}
      footer={<>
        <button className={BTN_GHOST} onClick={onClose}>{t("common.cancel")}</button>
        <button className={BTN_PRIMARY} disabled={busy || !name.trim()} onClick={submit}>
          {busy && <Loader2 className="h-4 w-4 animate-spin" />} {t("common.create")}
        </button>
      </>}>
      <Field label={t("kbview.name")}><input className={INPUT} value={name} onChange={(e) => setName(e.target.value)} placeholder={t("kbview.kbNamePlaceholder")} /></Field>
      <Field label={t("kbview.description")}><input className={INPUT} value={description} onChange={(e) => setDescription(e.target.value)} placeholder={t("common.optional")} /></Field>
      <Field label={t("kbview.visibility")}>
        <select className={INPUT} value={visibility} onChange={(e) => setVisibility(e.target.value)}>
          <option value="private">{t("kbview.visPrivate")}</option>
          <option value="workspace">{t("kbview.visWorkspace")}</option>
          <option value="public">public</option>
        </select>
      </Field>
      <div className="mt-1 grid grid-cols-2 gap-3">
        <Field label="chunk_size"><input type="number" className={cn(INPUT, "font-mono")} value={chunkSize} onChange={(e) => setChunkSize(Math.max(100, Number(e.target.value) || 0))} /></Field>
        <Field label="chunk_overlap"><input type="number" className={cn(INPUT, "font-mono")} value={chunkOverlap} onChange={(e) => setChunkOverlap(Math.max(0, Number(e.target.value) || 0))} /></Field>
      </div>
      <Field label={t("kbview.embeddingModelLabel")}>
        <select className={INPUT} value={embeddingModel} onChange={(e) => setEmbeddingModel(e.target.value)} disabled={!hasEmbedding}>
          <option value="">
            {hasEmbedding
              ? (cat.defaultEmbedding ? t("kbview.embDefaultWith", { model: cat.defaultEmbedding }) : t("kbview.embDefault"))
              : t("kbview.embLocalBuiltin")}
          </option>
          {cat.embedding.map((m) => (
            <option key={m.id} value={m.id}>{m.id}{m.dimension ? ` · ${m.dimension}d` : ""}</option>
          ))}
        </select>
        <p className="mt-1 text-[11px] text-[var(--text-faint)]">{t("kbview.embFixedNote2")}</p>
      </Field>
    </Drawer>
  );
}

// --- utils ----------------------------------------------------------------- //
function fmtTime(t: TFunction, iso?: string): string {
  if (!iso) return "—";
  const d = new Date(iso), diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return t("kbview.justNow");
  if (diff < 3600) return t("kbview.minutesAgo", { n: Math.floor(diff / 60) });
  if (diff < 86400) return t("kbview.hoursAgo", { n: Math.floor(diff / 3600) });
  return d.toLocaleDateString();
}

// Collapsed height ≈ 4 lines of 13px text at leading-relaxed (1.625).
const CHUNK_COLLAPSED_PX = 86;

// Chunk body with progressive disclosure: a masked 4-line preview that expands
// in place, revealing the heading breadcrumb / source offset and a copy action.
function ChunkBody({ text, query, headingPath, charStart, charEnd, className }: {
  text: string;
  query?: string;
  headingPath?: string[];
  charStart?: number | null;
  charEnd?: number | null;
  className?: string;
}) {
  const { t } = useI18n();
  const bodyRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false);
  const [overflows, setOverflows] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const el = bodyRef.current;
    if (el) setOverflows(el.scrollHeight > CHUNK_COLLAPSED_PX + 4);
  }, [text]);

  const copy = async () => {
    await copyText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };

  const clamped = !expanded && overflows;
  const showMeta = expanded && ((headingPath?.length ?? 0) > 0 || typeof charStart === "number");

  return (
    <div className={className}>
      {showMeta && (
        <div className="mb-1.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-[var(--text-faint)]">
          {headingPath?.length ? <span className="min-w-0 truncate">{headingPath.join(" › ")}</span> : null}
          {typeof charStart === "number" && (
            <span className="font-mono">{t("kbview.chunkRange", { start: charStart, end: charEnd ?? charStart })}</span>
          )}
        </div>
      )}
      <div
        ref={bodyRef}
        className="overflow-hidden transition-[max-height] duration-300 ease-out"
        style={{
          maxHeight: expanded ? bodyRef.current?.scrollHeight ?? 9999 : CHUNK_COLLAPSED_PX,
          maskImage: clamped ? "linear-gradient(to bottom, #000 58%, transparent)" : undefined,
          WebkitMaskImage: clamped ? "linear-gradient(to bottom, #000 58%, transparent)" : undefined,
        }}
      >
        <p className="whitespace-pre-wrap text-[13px] leading-relaxed text-[var(--text-muted)]">
          {query ? highlight(text, query) : text}
        </p>
      </div>
      {(overflows || expanded) && (
        <div className="mt-1.5 flex items-center gap-3.5 text-[11px] text-[var(--text-muted)]">
          <button
            className="inline-flex items-center gap-1 transition-colors hover:text-[var(--text)]"
            onClick={() => setExpanded((v) => !v)}
          >
            <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", expanded && "rotate-180")} />
            {expanded ? t("kbview.chunkCollapse") : t("kbview.chunkExpand")}
          </button>
          <button className="inline-flex items-center gap-1 transition-colors hover:text-[var(--text)]" onClick={copy}>
            {copied ? <Check className="h-3.5 w-3.5 text-[var(--success)]" /> : <Copy className="h-3.5 w-3.5" />}
            {copied ? t("common.copied") : t("common.copy")}
          </button>
        </div>
      )}
    </div>
  );
}

function highlight(text: string, query: string): ReactNode {
  const terms = query.toLowerCase().split(/\s+/).filter((term) => term.length > 1);
  if (!terms.length) return text;
  const re = new RegExp(`(${terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
  return text.split(re).map((p, i) =>
    terms.includes(p.toLowerCase())
      ? <mark key={i} className="rounded-[2px] bg-[var(--accent)]/25 px-0.5 text-[var(--text)]">{p}</mark>
      : <span key={i}>{p}</span>
  );
}
