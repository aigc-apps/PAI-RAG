import { useEffect, useState, type ReactNode } from "react";
import {
  ArrowLeft, Bot, Check, Copy, Database, Eye, Globe, Info, Loader2, Pencil, Plus,
  RefreshCw, Search, Trash2, TriangleAlert, Upload, X,
} from "lucide-react";
import { toast } from "sonner";
import {
  createDataSource, createKnowledgeBase, deleteDataSource, deleteKnowledgeBase,
  getSearchEngine, getUploadSupport, importKnowledgeDocument, listDataSources,
  listKnowledgeBases, listKnowledgeChunks, listKnowledgeDocuments, searchKnowledge,
  syncDataSource, updateDataSource, updateKnowledgeBase, uploadKnowledgeDocument,
  type KnowledgeBase, type KnowledgeBasePatch, type KnowledgeChunk,
  type KnowledgeDataSource, type KnowledgeDocument, type KnowledgeHit,
  type SearchEngineStatus,
} from "../api/knowledge";
import { cn } from "../lib/cn";
import { copyText } from "../lib/clipboard";

// --- shared class strings (match SettingsView / app idioms) ---------------- //
const CARD = "rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-elevated)] p-4";
const INPUT = "w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm outline-none focus:border-[var(--accent)]";
const BTN_PRIMARY = "inline-flex items-center justify-center gap-1.5 rounded-[var(--radius-sm)] bg-[var(--accent)] px-3 py-2 text-sm font-medium text-[var(--accent-fg)] hover:bg-[var(--accent-hover)] disabled:opacity-50";
const BTN_GHOST = "inline-flex items-center justify-center gap-1.5 rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-2 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] disabled:opacity-50";
const ICON_BTN = "grid h-8 w-8 place-items-center rounded-[var(--radius-sm)] text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] disabled:opacity-40 disabled:hover:bg-transparent";

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

const STATUS_ZH: Record<string, string> = {
  ready: "就绪", empty: "待索引", indexed: "已索引", active: "启用",
  processing: "处理中", failed: "失败", has_errors: "有错误", disabled: "已停用", deleted: "已删除",
  website: "website", upload: "upload", text: "text", file: "file",
  // data source sync states
  idle: "未同步", syncing: "同步中", succeeded: "已同步", partial: "部分成功",
  llms_txt: "阿里云文档",
};

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
  return (
    <span className={cn("rounded-full border px-2 py-0.5 text-[11px] font-medium whitespace-nowrap", statusClass(status))}>
      {STATUS_ZH[status] ?? status}
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
          <button type="button" aria-label="关闭" onClick={onClose} className={ICON_BTN}>
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
export function KnowledgeView({ onBack }: { onBack: () => void }) {
  const [bases, setBases] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [openId, setOpenId] = useState<string | null>(null);

  const refresh = async () => {
    const rows = await listKnowledgeBases();
    setBases(rows);
    return rows;
  };

  useEffect(() => {
    (async () => {
      setLoading(true);
      try { await refresh(); }
      catch (e) { toast.error(e instanceof Error ? e.message : "无法加载知识库"); }
      finally { setLoading(false); }
    })();
  }, []);

  const open = openId ? bases.find((b) => b.id === openId) ?? null : null;

  if (open) {
    return (
      <KbDetail
        kb={open}
        onBackToList={() => setOpenId(null)}
        onChanged={async () => { await refresh(); }}
        onDeleted={async () => { setOpenId(null); await refresh(); }}
      />
    );
  }

  return (
    <div className="flex h-full flex-col bg-[var(--bg)] text-[var(--text)]">
      <TopBar onBack={onBack} crumb={<b className="font-semibold text-[var(--text)]">知识库</b>} />
      <div className="min-h-0 flex-1 overflow-y-auto scrollbar-thin">
        <KbList
          bases={bases} loading={loading}
          onOpen={setOpenId}
          onCreated={async (id) => { await refresh(); setOpenId(id); }}
        />
      </div>
    </div>
  );
}

function TopBar({ onBack, crumb }: { onBack: () => void; crumb: ReactNode }) {
  return (
    <div className="flex h-12 flex-shrink-0 items-center gap-2 border-b border-[var(--border)] px-3">
      <button type="button" aria-label="返回" onClick={onBack} className={ICON_BTN}>
        <ArrowLeft className="h-4 w-4" />
      </button>
      <span className="grid h-6 w-6 place-items-center rounded-[6px] bg-[var(--accent-soft)] text-[var(--accent)]">
        <Database className="h-3.5 w-3.5" />
      </span>
      <div className="text-[13px] text-[var(--text-muted)]">{crumb}</div>
    </div>
  );
}

// ======================================================================== //
// List
// ======================================================================== //
function KbList({ bases, loading, onOpen, onCreated }: {
  bases: KnowledgeBase[]; loading: boolean;
  onOpen: (id: string) => void; onCreated: (id: string) => void;
}) {
  const [showCreate, setShowCreate] = useState(false);

  return (
    <div className="mx-auto w-full max-w-[1120px] px-5 pt-6 pb-16">
      <div className="mb-5 flex items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">知识库</h1>
          <p className="mt-1 max-w-2xl text-[13px] text-[var(--text-muted)]">
            为智能体提供可检索、可追溯的私有知识。每个知识库有独立的向量引擎、Embedding 与切片配置。
          </p>
        </div>
        <button className={BTN_PRIMARY} onClick={() => setShowCreate(true)}>
          <Plus className="h-4 w-4" /> 新建知识库
        </button>
      </div>

      {loading ? (
        <div className="flex items-center gap-2 py-10 text-sm text-[var(--text-muted)]">
          <Loader2 className="h-4 w-4 animate-spin" /> 加载中
        </div>
      ) : bases.length === 0 ? (
        <div className="grid place-items-center rounded-[var(--radius-lg)] border border-dashed border-[var(--border)] py-16 text-center">
          <Database className="h-6 w-6 text-[var(--text-faint)]" />
          <div className="mt-3 text-sm text-[var(--text-muted)]">暂无知识库</div>
          <button className={cn(BTN_GHOST, "mt-3")} onClick={() => setShowCreate(true)}>
            <Plus className="h-4 w-4" /> 创建第一个
          </button>
        </div>
      ) : (
        <div className="grid gap-3.5" style={{ gridTemplateColumns: "repeat(auto-fill,minmax(320px,1fr))" }}>
          {bases.map((kb) => <KbCard key={kb.id} kb={kb} onOpen={() => onOpen(kb.id)} />)}
        </div>
      )}

      <CreateDrawer
        open={showCreate}
        onClose={() => setShowCreate(false)}
        onCreated={(id) => { setShowCreate(false); onCreated(id); }}
      />
    </div>
  );
}

function KbCard({ kb, onOpen }: { kb: KnowledgeBase; onOpen: () => void }) {
  const emb = kb.embedding_config ?? {};
  const parser = parserOf(kb);
  const rerank = kb.rerank_config?.enabled;
  return (
    <button
      type="button" onClick={onOpen}
      className="flex flex-col gap-3 rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-elevated)] p-4 text-left transition-colors hover:border-[var(--border-strong)] hover:shadow-[var(--shadow)]"
    >
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-[15px] font-semibold">{kb.name}</h3>
        <Pill status={kb.status} />
      </div>
      <div className="min-h-[18px] truncate text-[12.5px] text-[var(--text-muted)]">
        {kb.description || "无描述"}
      </div>
      <div className="flex items-center gap-3.5 text-xs text-[var(--text-muted)]">
        <span><b className="font-semibold text-[var(--text)]">{kb.document_count}</b> 文档</span>
        <span><b className="font-semibold text-[var(--text)]">{kb.chunk_count}</b> 切片</span>
        <span className="rounded-full border border-[var(--border)] bg-[var(--surface)] px-2 py-0.5">{kb.visibility}</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        <Chip mono>{str(emb.model, "local-hash-v1")} · {num(emb.dimension, 64)}d</Chip>
        <Chip>向量 <span className="text-[var(--text-faint)]">{str(kb.vector_store_config?.provider_id, "local_sql")}</span></Chip>
        <Chip>重排 <span className="text-[var(--text-faint)]">{rerank ? str(kb.rerank_config?.model, "on") : "关闭"}</span></Chip>
        <Chip>切片 <span className="text-[var(--text-faint)]">{parser.chunk_size}/{parser.chunk_overlap}</span></Chip>
      </div>
    </button>
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
type Tab = "overview" | "config" | "datasources" | "files" | "recall";

function KbDetail({ kb, onBackToList, onChanged, onDeleted }: {
  kb: KnowledgeBase; onBackToList: () => void;
  onChanged: () => Promise<void>; onDeleted: () => Promise<void>;
}) {
  const [tab, setTab] = useState<Tab>("overview");

  const tabs: { id: Tab; label: string; badge?: number }[] = [
    { id: "overview", label: "概览" },
    { id: "config", label: "配置" },
    { id: "datasources", label: "数据源" },
    { id: "files", label: "文档", badge: kb.document_count },
    { id: "recall", label: "召回测试" },
  ];

  const remove = async () => {
    if (!confirm(`删除知识库「${kb.name}」？该操作不可撤销。`)) return;
    try { await deleteKnowledgeBase(kb.id); toast.success("已删除"); await onDeleted(); }
    catch (e) { toast.error(e instanceof Error ? e.message : "删除失败"); }
  };

  return (
    <div className="flex h-full flex-col bg-[var(--bg)] text-[var(--text)]">
      <TopBar
        onBack={onBackToList}
        crumb={<>
          <span className="cursor-pointer hover:text-[var(--text)]" onClick={onBackToList}>知识库</span>
          <span className="mx-1.5 text-[var(--text-faint)]">/</span>
          <b className="font-semibold text-[var(--text)]">{kb.name}</b>
        </>}
      />
      <div className="flex flex-shrink-0 items-center gap-1 border-b border-[var(--border)] px-3">
        {tabs.map((t) => (
          <button
            key={t.id} type="button" onClick={() => setTab(t.id)}
            className={cn(
              "-mb-px border-b-2 px-3.5 py-2.5 text-[13px] font-medium",
              tab === t.id
                ? "border-[var(--accent)] text-[var(--text)]"
                : "border-transparent text-[var(--text-muted)] hover:text-[var(--text)]"
            )}
          >
            {t.label}{typeof t.badge === "number" && <span className="ml-1 text-[var(--text-faint)]">{t.badge}</span>}
          </button>
        ))}
        <div className="flex-1" />
        <button type="button" onClick={remove} className={cn(ICON_BTN, "hover:text-[var(--danger)]")} title="删除知识库">
          <Trash2 className="h-4 w-4" />
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto scrollbar-thin">
        <div className="mx-auto w-full max-w-[1120px] px-5 py-6">
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
    <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] px-4 py-3">
      <div className="text-[22px] font-semibold tracking-tight">{n}</div>
      <div className="mt-0.5 text-[11.5px] text-[var(--text-faint)]">{l}</div>
    </div>
  );
}

function KV({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="flex justify-between gap-3 border-b border-[var(--border)] py-1.5 text-[13px] last:border-b-0">
      <span className="text-[var(--text-muted)]">{k}</span>
      <span className="font-mono text-xs">{v}</span>
    </div>
  );
}

function OverviewPanel({ kb }: { kb: KnowledgeBase }) {
  const emb = kb.embedding_config ?? {};
  const parser = parserOf(kb);
  const ret = retrievalOf(kb);
  const rr = kb.rerank_config ?? {};
  return (
    <div className="space-y-3.5">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat n={kb.document_count} l="文档" />
        <Stat n={kb.chunk_count} l="切片 (active)" />
        <Stat n={STATUS_ZH[kb.status] ?? kb.status} l="状态" />
        <Stat n={kb.visibility} l="可见性" />
      </div>
      <div className={CARD}>
        <div className="mb-1 flex items-center gap-2">
          <h3 className="text-sm font-semibold">配置摘要</h3>
          <span className="ml-auto inline-flex items-center gap-1.5 text-[11px] text-[var(--text-muted)]">
            <Info className="h-3 w-3" /> 模型来自 Providers
          </span>
        </div>
        <div className="mt-3 grid gap-x-5 md:grid-cols-2">
          <div>
            <KV k="Embedding" v={`${str(emb.provider_id, "local_hash")} / ${str(emb.model, "local-hash-v1")}`} />
            <KV k="向量维度" v={num(emb.dimension, 64)} />
            <KV k="向量引擎" v={`${str(kb.vector_store_config?.provider_id, "local_sql")} · ${str(kb.vector_store_config?.metric, "cosine")}`} />
          </div>
          <div>
            <KV k="Reranker" v={rr.enabled ? `${str(rr.provider_id, "-")} / ${str(rr.model, "-")}` : "关闭"} />
            <KV k="切片" v={`${parser.chunk_size} / overlap ${parser.chunk_overlap}`} />
            <KV k="检索模式" v={`${ret.mode} · top_k ${ret.top_k}`} />
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------- Config ---------- //
function ConfigPanel({ kb, onSaved }: { kb: KnowledgeBase; onSaved: () => Promise<void> }) {
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
      toast.success("配置已保存");
      await onSaved();
    } catch (e) { toast.error(e instanceof Error ? e.message : "保存失败"); }
    finally { setSaving(false); }
  };

  return (
    <div className="space-y-3.5">
      {/* basic */}
      <div className={CARD}>
        <h3 className="text-sm font-semibold">基本信息</h3>
        <div className="mt-3 grid gap-3.5 md:grid-cols-2">
          <Field label="名称"><input className={INPUT} value={name} onChange={(e) => setName(e.target.value)} /></Field>
          <Field label="可见性">
            <select className={INPUT} value={visibility} onChange={(e) => setVisibility(e.target.value)}>
              <option value="private">private — 仅自己</option>
              <option value="workspace">workspace — 团队可查询</option>
              <option value="public">public</option>
            </select>
          </Field>
          <div className="md:col-span-2">
            <Field label="描述"><input className={INPUT} value={description} onChange={(e) => setDescription(e.target.value)} placeholder="可选" /></Field>
          </div>
        </div>
      </div>

      {/* provider-sourced, read-only this phase */}
      <div className={CARD}>
        <div className="mb-1 flex items-center gap-2">
          <h3 className="text-sm font-semibold">Embedding · 向量引擎 · Reranker</h3>
          <span className="ml-auto rounded-full border border-[var(--border)] bg-[var(--surface)] px-2 py-0.5 text-[11px] text-[var(--text-muted)]">接入 Provider 后可选</span>
        </div>
        <div className="mt-3 grid gap-x-5 md:grid-cols-2">
          <div>
            <KV k="Embedding provider" v={str(emb.provider_id, "local_hash")} />
            <KV k="Embedding 模型" v={str(emb.model, "local-hash-v1")} />
            <KV k="向量维度" v={num(emb.dimension, 64)} />
          </div>
          <div>
            <KV k="向量引擎" v={str(vec.provider_id, "local_sql")} />
            <KV k="距离度量" v={str(vec.metric, "cosine")} />
            <KV k="Reranker" v={kb.rerank_config?.enabled ? "启用" : "关闭"} />
          </div>
        </div>
        <div className="mt-3 flex items-start gap-2 rounded-[var(--radius)] border border-[var(--accent)]/30 bg-[var(--accent-soft)] px-3 py-2.5 text-[12.5px] text-[var(--accent)]">
          <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
          <div>当前使用内置 <b>local</b> 引擎。在 <b>设置 → Providers</b> 配置 embedding / rerank / vectordb 类型的 Provider 后，这里将可切换外部模型（切换需重建索引）。</div>
        </div>
      </div>

      {/* chunking */}
      <div className={CARD}>
        <h3 className="text-sm font-semibold">切片</h3>
        <p className="mt-0.5 text-xs text-[var(--text-faint)]">影响之后导入的文档；已索引文档需重建索引后生效。</p>
        <div className="mt-3 grid gap-3.5 md:grid-cols-2">
          <Field label="切片大小 chunk_size（字符）">
            <input type="number" className={cn(INPUT, "font-mono")} value={chunkSize}
              onChange={(e) => setChunkSize(Math.max(100, Number(e.target.value) || 0))} />
          </Field>
          <Field label="重叠 chunk_overlap">
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
          <span>切片 N</span><span className="text-[var(--accent)]">重叠 {chunkOverlap}</span><span>切片 N+1</span>
        </div>
      </div>

      {/* retrieval */}
      <div className={CARD}>
        <h3 className="text-sm font-semibold">检索默认值</h3>
        <p className="mt-0.5 text-xs text-[var(--text-faint)]">召回测试与智能体查询的初始参数。</p>
        <div className="mt-3 flex flex-wrap items-end gap-5">
          <Field label="模式">
            <Seg value={mode} onChange={setMode} options={[
              { value: "hybrid", label: "hybrid" }, { value: "vector", label: "vector" }, { value: "keyword", label: "keyword" },
            ]} />
          </Field>
          <div className="w-24"><Field label="top_k">
            <input type="number" className={cn(INPUT, "font-mono")} value={topK}
              onChange={(e) => setTopK(Math.max(1, Number(e.target.value) || 1))} />
          </Field></div>
          <div className="w-28"><Field label="分数阈值">
            <input type="number" step="0.05" className={cn(INPUT, "font-mono")} value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value) || 0)} />
          </Field></div>
          <div className="pb-2"><Toggle on={forceCite} onChange={setForceCite} label="强制引用来源" /></div>
        </div>
      </div>

      {indexAffecting && (
        <div className="flex items-start gap-2.5 rounded-[var(--radius)] border border-[var(--warning)]/40 bg-[var(--warning)]/10 px-3.5 py-3 text-[12.5px] text-[var(--warning)]">
          <TriangleAlert className="mt-0.5 h-4 w-4 flex-shrink-0" />
          <div>切片参数已更改，将影响已索引的 {kb.chunk_count} 个切片；保存后需 <b>重建索引</b> 才会对旧文档生效（重建功能即将上线）。</div>
        </div>
      )}

      <div className="sticky bottom-0 flex items-center gap-3 rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-3 shadow-[var(--shadow)]">
        {dirty
          ? <span className="flex items-center gap-2 text-[12.5px] text-[var(--warning)]"><span className="h-1.5 w-1.5 rounded-full bg-[var(--warning)]" /> 有未保存的更改</span>
          : <span className="text-[12.5px] text-[var(--text-faint)]">已是最新</span>}
        <div className="flex-1" />
        <button className={BTN_GHOST} disabled={!dirty || saving} onClick={reset}>放弃</button>
        <button className={BTN_PRIMARY} disabled={!dirty || saving} onClick={save}>
          {saving && <Loader2 className="h-4 w-4 animate-spin" />} 保存配置
        </button>
      </div>
    </div>
  );
}

// ---------- Data sources ---------- //
function sourceSummary(ds: KnowledgeDataSource): string {
  const cfg = ds.source_config || {};
  if (typeof cfg.llms_url === "string" && cfg.llms_url) return cfg.llms_url;
  if (typeof cfg.product === "string" && cfg.product)
    return `help.aliyun.com/zh/${cfg.product}/llms.txt`;
  return "—";
}

function DataSourcePanel({ kb, onChanged }: { kb: KnowledgeBase; onChanged: () => Promise<void> }) {
  const [sources, setSources] = useState<KnowledgeDataSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState<KnowledgeDataSource | "new" | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const refresh = async () => {
    try { setSources(await listDataSources(kb.id)); }
    catch (e) { toast.error(e instanceof Error ? e.message : "无法加载数据源"); }
    finally { setLoading(false); }
  };
  useEffect(() => { setLoading(true); void refresh(); }, [kb.id]);

  // poll while any source is syncing; also refresh the KB counts when it settles
  useEffect(() => {
    if (!sources.some((s) => s.status === "syncing")) return;
    const t = setTimeout(async () => { await refresh(); void onChanged(); }, 2500);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sources]);

  const runSync = async (ds: KnowledgeDataSource) => {
    setBusyId(ds.id);
    try {
      await syncDataSource(kb.id, ds.id);
      toast.success("已开始同步，正在拉取文档…");
      await refresh();
    } catch (e) { toast.error(e instanceof Error ? e.message : "同步失败"); }
    finally { setBusyId(null); }
  };

  const toggleEnabled = async (ds: KnowledgeDataSource) => {
    try { await updateDataSource(kb.id, ds.id, { enabled: !ds.enabled }); await refresh(); }
    catch (e) { toast.error(e instanceof Error ? e.message : "更新失败"); }
  };

  const remove = async (ds: KnowledgeDataSource) => {
    if (!confirm(`删除数据源「${ds.name}」？已导入的文档将保留。`)) return;
    try { await deleteDataSource(kb.id, ds.id); toast.success("已删除"); await refresh(); }
    catch (e) { toast.error(e instanceof Error ? e.message : "删除失败"); }
  };

  return (
    <div className="space-y-3.5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <p className="max-w-2xl text-[13px] text-[var(--text-muted)]">
          从外部来源自动拉取文档并入库。目前支持<b className="text-[var(--text)]">阿里云帮助文档</b>（
          <code className="font-mono text-[11px]">llms.txt</code> 清单）：填入产品标识即可同步该产品的官方文档，重复同步只更新变化的内容。
        </p>
        <button className={BTN_PRIMARY} onClick={() => setEditing("new")}>
          <Plus className="h-4 w-4" /> 添加数据源
        </button>
      </div>

      {loading ? (
        <div className="flex items-center gap-2 py-10 text-sm text-[var(--text-muted)]">
          <Loader2 className="h-4 w-4 animate-spin" /> 加载中
        </div>
      ) : sources.length === 0 ? (
        <div className="grid place-items-center rounded-[var(--radius-lg)] border border-dashed border-[var(--border)] py-16 text-center">
          <Globe className="h-6 w-6 text-[var(--text-faint)]" />
          <div className="mt-3 text-sm text-[var(--text-muted)]">还没有数据源</div>
          <button className={cn(BTN_GHOST, "mt-3")} onClick={() => setEditing("new")}>
            <Plus className="h-4 w-4" /> 添加阿里云文档数据源
          </button>
        </div>
      ) : (
        <div className="grid gap-3">
          {sources.map((ds) => {
            const syncing = ds.status === "syncing";
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
                      {!ds.enabled && <span className="text-[11px] text-[var(--text-faint)]">已停用</span>}
                    </div>
                    <div className="mt-0.5 truncate font-mono text-[11px] text-[var(--text-faint)]">{sourceSummary(ds)}</div>
                  </div>
                  <Pill status={ds.status} />
                </div>

                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 pl-[38px] text-[12px] text-[var(--text-muted)]">
                  <span><b className="font-semibold text-[var(--text)] tabular-nums">{ds.doc_count}</b> 文档</span>
                  <span>最近同步 {fmtTime(ds.last_sync_finished_at || ds.last_sync_at || undefined)}</span>
                  {(ds.status === "succeeded" || ds.status === "partial") && (
                    <span className="font-mono text-[11px] text-[var(--text-faint)]">
                      +{rep.added ?? 0} 新增 · {rep.updated ?? 0} 更新 · {rep.deleted ?? 0} 删除
                      {typeof rep.failed === "number" && rep.failed > 0 ? ` · ${rep.failed} 失败` : ""}
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
                  <Toggle on={ds.enabled} onChange={() => void toggleEnabled(ds)} label={ds.enabled ? "已启用" : "已停用"} />
                  <div className="flex-1" />
                  <button
                    className={BTN_GHOST} disabled={syncing || busyId === ds.id || !ds.enabled}
                    onClick={() => void runSync(ds)} title={ds.enabled ? "立即同步" : "启用后可同步"}
                  >
                    <RefreshCw className={cn("h-4 w-4", (syncing || busyId === ds.id) && "animate-spin")} />
                    {syncing ? "同步中" : "同步"}
                  </button>
                  <button className={ICON_BTN} title="编辑" onClick={() => setEditing(ds)}><Pencil className="h-4 w-4" /></button>
                  <button className={cn(ICON_BTN, "hover:text-[var(--danger)]")} title="删除" onClick={() => void remove(ds)}><Trash2 className="h-4 w-4" /></button>
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
  const open = editing !== null;
  const isEdit = editing !== null && editing !== "new";
  const src = isEdit ? editing.source_config || {} : {};
  const initUrl = typeof src.llms_url === "string" ? src.llms_url : "";

  const [name, setName] = useState(isEdit ? editing.name : "");
  const [mode, setMode] = useState<"product" | "url">(initUrl ? "url" : "product");
  const [product, setProduct] = useState(typeof src.product === "string" ? src.product : "");
  const [llmsUrl, setLlmsUrl] = useState(initUrl);
  const [sections, setSections] = useState(Array.isArray(src.sections) ? (src.sections as string[]).join(", ") : "");
  const [lang, setLang] = useState(typeof src.lang === "string" ? src.lang : "zh");
  const [busy, setBusy] = useState(false);

  const valid = name.trim() !== "" && (mode === "product" ? product.trim() !== "" : llmsUrl.trim() !== "");

  const submit = async () => {
    if (!valid) return;
    setBusy(true);
    try {
      const cfg: Record<string, unknown> = {};
      if (mode === "product") cfg.product = product.trim();
      else cfg.llms_url = llmsUrl.trim();
      const secs = sections.split(",").map((s) => s.trim()).filter(Boolean);
      if (secs.length) cfg.sections = secs;
      cfg.lang = lang.trim() || "zh";
      if (isEdit) {
        await updateDataSource(kb.id, editing.id, { name: name.trim(), source_config: cfg });
        toast.success("数据源已更新");
      } else {
        await createDataSource(kb.id, { name: name.trim(), source_type: "llms_txt", source_config: cfg });
        toast.success("数据源已添加，点击「同步」拉取文档");
      }
      await onDone();
    } catch (e) { toast.error(e instanceof Error ? e.message : "保存失败"); }
    finally { setBusy(false); }
  };

  return (
    <Drawer open={open} title={isEdit ? "编辑数据源" : "添加数据源"} onClose={onClose}
      footer={<>
        <button className={BTN_GHOST} onClick={onClose}>取消</button>
        <button className={BTN_PRIMARY} disabled={busy || !valid} onClick={submit}>
          {busy && <Loader2 className="h-4 w-4 animate-spin" />} {isEdit ? "保存" : "添加"}
        </button>
      </>}>
      <Field label="类型">
        <div className="flex items-center gap-2 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-[13px]">
          <Globe className="h-4 w-4 text-[var(--accent)]" /> 阿里云帮助文档 (llms.txt)
        </div>
      </Field>
      <Field label="名称"><input className={INPUT} value={name} onChange={(e) => setName(e.target.value)} placeholder="如：PAI 官方文档" /></Field>

      <Field label="来源方式">
        <Seg value={mode} onChange={setMode} options={[
          { value: "product", label: "产品标识" }, { value: "url", label: "完整 URL" },
        ]} />
      </Field>
      {mode === "product" ? (
        <Field label="产品标识 product" hint="help.aliyun.com/zh/<product>/ 中的 product 段，如 pai、eas、oss">
          <input className={cn(INPUT, "font-mono text-xs")} value={product}
            onChange={(e) => setProduct(e.target.value)} placeholder="pai" />
        </Field>
      ) : (
        <Field label="llms.txt URL" hint="子产品或非标准路径时使用完整清单地址">
          <input className={cn(INPUT, "font-mono text-xs")} value={llmsUrl}
            onChange={(e) => setLlmsUrl(e.target.value)} placeholder="https://help.aliyun.com/zh/pai/llms.txt" />
        </Field>
      )}
      <Field label="章节过滤 sections" hint="可选，逗号分隔；留空同步全部章节">
        <input className={INPUT} value={sections} onChange={(e) => setSections(e.target.value)} placeholder="快速开始, 最佳实践" />
      </Field>
      <Field label="语言 lang" hint="记录在文档元数据上，默认 zh">
        <input className={cn(INPUT, "w-24")} value={lang} onChange={(e) => setLang(e.target.value)} placeholder="zh" />
      </Field>
      <div className="flex items-start gap-2 rounded-[var(--radius)] border border-[var(--accent)]/30 bg-[var(--accent-soft)] px-3 py-2.5 text-[12px] text-[var(--accent)]">
        <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
        <div>保存后在列表点击「同步」拉取文档。仅抓取公开的 llms.txt 清单与官方 <code className="font-mono">.md</code> 正文，按当前切片配置入库。</div>
      </div>
    </Drawer>
  );
}

// ---------- Files ---------- //
function FilesPanel({ kb, onChanged }: { kb: KnowledgeBase; onChanged: () => Promise<void> }) {
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
    const t = setTimeout(() => { void load({ append: false, query: q }); }, q ? 250 : 0);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kb.id, q]);

  const refresh = () => load({ append: false });
  const hasMore = docs.length < total;

  // poll while any doc is still processing (queued/running ingest job); refresh
  // the KB counts once it settles to indexed. Mirrors the datasource poller.
  useEffect(() => {
    if (!docs.some((d) => d.status === "processing")) return;
    const t = setTimeout(async () => { await refresh(); void onChanged(); }, 2500);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docs]);

  return (
    <div className="space-y-3.5">
      <div className="flex flex-wrap items-center gap-2.5">
        <div className="relative max-w-[320px] flex-1">
          <Search className="absolute top-1/2 left-2.5 h-3.5 w-3.5 -translate-y-1/2 text-[var(--text-faint)]" />
          <input className={cn(INPUT, "pl-8")} placeholder="搜索标题 / URL" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <div className="flex-1" />
        {!loading && <span className="text-[12px] text-[var(--text-faint)] tabular-nums">共 {total} 篇</span>}
        <button className={BTN_PRIMARY} onClick={() => setShowImport(true)}><Plus className="h-4 w-4" /> 导入文档</button>
      </div>

      <div className={cn(CARD, "p-1")}>
        {loading ? (
          <div className="flex items-center gap-2 p-6 text-sm text-[var(--text-muted)]"><Loader2 className="h-4 w-4 animate-spin" /> 加载中</div>
        ) : docs.length === 0 ? (
          <div className="p-8 text-center text-sm text-[var(--text-muted)]">{q ? "无匹配文档" : "暂无文档，点击「导入文档」开始"}</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-[13px]">
              <thead>
                <tr className="text-left text-[11px] font-semibold tracking-wide text-[var(--text-faint)] uppercase">
                  <th className="border-b border-[var(--border)] px-3 py-2">标题</th>
                  <th className="border-b border-[var(--border)] px-3 py-2">来源</th>
                  <th className="border-b border-[var(--border)] px-3 py-2">状态</th>
                  <th className="border-b border-[var(--border)] px-3 py-2 text-right">切片</th>
                  <th className="border-b border-[var(--border)] px-3 py-2">标签</th>
                  <th className="border-b border-[var(--border)] px-3 py-2 text-right">索引时间</th>
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
                        {(d.tags || []).slice(0, 3).map((t) => (
                          <span key={t} className="rounded-full bg-[var(--surface-2)] px-1.5 py-0.5 text-[10.5px] text-[var(--text-muted)]">{t}</span>
                        ))}
                      </div>
                    </td>
                    <td className="border-b border-[var(--border)] px-3 py-2.5 text-right text-[var(--text-faint)]">{fmtTime(d.indexed_at)}</td>
                    <td className="border-b border-[var(--border)] px-3 py-2.5">
                      <div className="flex justify-end gap-1 opacity-60 group-hover:opacity-100">
                        <button className={ICON_BTN} title="查看切片" onClick={() => setViewDoc(d)}><Eye className="h-4 w-4" /></button>
                        <button className={ICON_BTN} title="重新索引（即将上线）" disabled><RefreshCw className="h-4 w-4" /></button>
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
              加载更多（{docs.length}/{total}）
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
    <Drawer open={!!doc} title={doc ? `切片 · ${doc.title}（${total}）` : "切片"} onClose={onClose}
      footer={<button className={BTN_GHOST} onClick={onClose}>关闭</button>}>
      {loading ? (
        <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]"><Loader2 className="h-4 w-4 animate-spin" /> 加载中</div>
      ) : chunks.length === 0 ? (
        <div className="text-sm text-[var(--text-muted)]">暂无切片</div>
      ) : (
        <div className="space-y-2">
          {chunks.map((c) => (
            <div key={c.id} className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-3">
              <div className="mb-1 flex justify-between text-[11px] text-[var(--text-muted)]">
                <span>#{c.chunk_index} · {c.token_count} tokens</span><Pill status={c.status} />
              </div>
              <p className="text-[13px] leading-relaxed line-clamp-4">{c.text}</p>
            </div>
          ))}
          {hasMore && (
            <div className="flex justify-center pt-1">
              <button className={BTN_GHOST} disabled={loadingMore} onClick={() => void load(true)}>
                {loadingMore ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                加载更多（{chunks.length}/{total}）
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
        tags: form.tags.split(",").map((t) => t.trim()).filter(Boolean),
      });
      setForm({ title: "", uri: "", tags: "", content: "" });
      toast.success("已导入并索引");
      await onDone();
    } catch (e) { toast.error(e instanceof Error ? e.message : "导入失败"); }
    finally { setBusy(false); }
  };
  const upload = async (file: File) => {
    if (file.size > maxMb * 1024 * 1024) {
      toast.error(`文件超过 ${maxMb}MB 上限`);
      return;
    }
    setBusy(true);
    try {
      await uploadKnowledgeDocument(kb.id, file, {
        title: form.title.trim() || undefined,
        tags: form.tags.split(",").map((t) => t.trim()).filter(Boolean),
      });
      toast.success("已上传并索引");
      await onDone();
    } catch (e) { toast.error(e instanceof Error ? e.message : "上传失败"); }
    finally { setBusy(false); }
  };
  return (
    <Drawer open={open} title="导入文档" onClose={onClose}
      footer={<>
        <button className={BTN_GHOST} onClick={onClose}>取消</button>
        <button className={BTN_PRIMARY} disabled={busy || !form.title.trim() || !form.content.trim()} onClick={submit}>
          {busy && <Loader2 className="h-4 w-4 animate-spin" />} 导入并索引
        </button>
      </>}>
      <Field label="上传文件" hint={accept ? `支持 ${accept}（≤${maxMb}MB）` : "PDF / Word / PPT / Excel / 文本等"}>
        <label className={cn(BTN_GHOST, "w-full cursor-pointer")}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
          选择文件上传
          <input type="file" accept={accept || undefined} className="hidden" disabled={busy}
            onChange={(e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (f) void upload(f);
            }} />
        </label>
      </Field>
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-[var(--text-muted)]">
        <div className="h-px flex-1 bg-[var(--border)]" />或手动粘贴<div className="h-px flex-1 bg-[var(--border)]" />
      </div>
      <Field label="标题"><input className={INPUT} value={form.title} onChange={(e) => setForm((s) => ({ ...s, title: e.target.value }))} placeholder="文档标题" /></Field>
      <Field label="来源 URL / OSS URI" hint="可选，留空作为纯文本">
        <input className={cn(INPUT, "font-mono text-xs")} value={form.uri} onChange={(e) => setForm((s) => ({ ...s, uri: e.target.value }))} placeholder="https://…" />
      </Field>
      <Field label="标签（逗号分隔）"><input className={INPUT} value={form.tags} onChange={(e) => setForm((s) => ({ ...s, tags: e.target.value }))} placeholder="eas, product-docs" /></Field>
      <Field label="内容">
        <textarea rows={9} className={cn(INPUT, "resize-none")} value={form.content}
          onChange={(e) => setForm((s) => ({ ...s, content: e.target.value }))}
          placeholder="粘贴文本内容。MVP 先做同步文本导入；网站抓取 / OSS 下载 / 文件解析后续接 ingestion adapter。" />
      </Field>
      <div className="flex items-start gap-2 rounded-[var(--radius)] border border-[var(--accent)]/30 bg-[var(--accent-soft)] px-3 py-2.5 text-[12px] text-[var(--accent)]">
        <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
        <div>按当前切片配置索引；相同 URL 会覆盖旧文档。</div>
      </div>
    </Drawer>
  );
}

// ---------- Recall ---------- //
// Ties the recall surface to the agent goal: tells the tester that a chat agent
// can retrieve from this KB via the knowledge_search tool, and hands over the id
// for scoping a call. Visibility gates whether other users' agents can reach it.
function AgentAvailability({ kb, queryable }: { kb: KnowledgeBase; queryable: boolean }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    if (await copyText(kb.id)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } else {
      toast.error("复制失败");
    }
  };
  return (
    <div className={cn(CARD, "flex flex-wrap items-center gap-x-4 gap-y-2.5")}>
      <span className="grid h-8 w-8 flex-shrink-0 place-items-center rounded-[var(--radius-sm)] bg-[var(--accent-soft)] text-[var(--accent)]">
        <Bot className="h-4 w-4" />
      </span>
      <div className="min-w-0 flex-1">
        <div className="text-[13px] font-semibold">智能体问答已接通</div>
        <div className="mt-0.5 text-[12px] text-[var(--text-muted)]">
          对话中的智能体可通过 <code className="font-mono text-[11px]">knowledge_search</code> 工具检索本知识库
          {queryable
            ? "。当前可见性允许团队智能体查询。"
            : "（当前为 private，仅所有者/管理员的智能体可查询）。"}
        </div>
      </div>
      <button
        className={cn(BTN_GHOST, "font-mono text-[11px]")} onClick={copy}
        title="复制知识库 ID，用于 knowledge_search 的 kb_ids 参数"
      >
        {copied ? <Check className="h-3.5 w-3.5 text-[var(--success)]" /> : <Copy className="h-3.5 w-3.5" />}
        {copied ? "已复制" : `id: ${kb.id.slice(0, 8)}…`}
      </button>
    </div>
  );
}

function EngineBadge({ engine }: { engine: SearchEngineStatus | null }) {
  if (!engine) return null;
  const isEs = engine.engine === "elasticsearch";
  const ok = !engine.configured || engine.healthy;
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full border border-[var(--border)] bg-[var(--surface)] px-2.5 py-1 text-[11.5px] text-[var(--text-muted)]"
      title={
        isEs
          ? engine.healthy ? "Elasticsearch 混合检索（BM25 + 向量 kNN）已连接" : "已配置 Elasticsearch 但当前不可达，自动降级本地检索"
          : "内置本地检索引擎（配置 ELASTICSEARCH_URL 可启用 Elasticsearch 混合检索）"
      }
    >
      <Database className={cn("h-3.5 w-3.5", ok ? "text-[var(--accent)]" : "text-[var(--warning,#d97706)]")} />
      {isEs ? "Elasticsearch" : "本地检索"}
      {engine.configured && !engine.healthy && <span className="text-[var(--warning,#d97706)]">· 不可达</span>}
    </span>
  );
}

function RecallPanel({ kb }: { kb: KnowledgeBase }) {
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
  const [engine, setEngine] = useState<SearchEngineStatus | null>(null);
  const [lastQuery, setLastQuery] = useState("");

  useEffect(() => { getSearchEngine().then(setEngine).catch(() => setEngine(null)); }, []);

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
    } catch (e) { toast.error(e instanceof Error ? e.message : "检索失败"); }
    finally { setBusy(false); }
  };

  const loadMore = async () => {
    if (!hits) return;
    setLoadingMore(true);
    try {
      const page = await fetchPage(hits.length);
      setHits((prev) => [...(prev || []), ...page.data]);
      setTotal(page.total);
    } catch (e) { toast.error(e instanceof Error ? e.message : "加载失败"); }
    finally { setLoadingMore(false); }
  };

  const queryable = kb.visibility === "workspace" || kb.visibility === "public";
  const hasMore = hits !== null && hits.length < total;

  return (
    <div className="space-y-3.5">
      <AgentAvailability kb={kb} queryable={queryable} />
      <div className="flex items-center gap-2">
        <span className="text-[12px] text-[var(--text-faint)]">检索引擎</span>
        <EngineBadge engine={engine} />
      </div>
      <div className={CARD}>
        <div className="flex gap-2">
          <input
            className={cn(INPUT, "flex-1")} value={query} placeholder="输入查询，测试召回效果…"
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") void run(); }}
          />
          <button className={BTN_PRIMARY} disabled={busy || !query.trim()} onClick={run}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />} 检索
          </button>
        </div>
        <div className="mt-3.5 flex flex-wrap items-center gap-x-5 gap-y-3 border-t border-[var(--border)] pt-3.5 text-[12.5px] text-[var(--text-muted)]">
          <span className="flex items-center gap-2">模式
            <Seg value={mode} onChange={setMode} options={[
              { value: "hybrid", label: "hybrid" }, { value: "vector", label: "vector" }, { value: "keyword", label: "keyword" },
            ]} />
          </span>
          <span className="flex items-center gap-2">top_k
            <input type="number" className={cn(INPUT, "w-16 px-2 py-1 font-mono")} value={topK} onChange={(e) => setTopK(Math.max(1, Number(e.target.value) || 1))} />
          </span>
          <span className="flex items-center gap-2">阈值
            <input type="number" step="0.05" className={cn(INPUT, "w-20 px-2 py-1 font-mono")} value={threshold} onChange={(e) => setThreshold(Number(e.target.value) || 0)} />
          </span>
          <span className="flex items-center gap-2">标签
            <input className={cn(INPUT, "w-28 px-2 py-1")} value={tag} onChange={(e) => setTag(e.target.value)} placeholder="可选" />
          </span>
        </div>
      </div>

      {hits === null ? (
        <div className="grid place-items-center rounded-[var(--radius-lg)] border border-dashed border-[var(--border)] py-14 text-center">
          <Search className="h-6 w-6 text-[var(--text-faint)]" />
          <div className="mt-3 text-sm text-[var(--text-muted)]">输入查询并检索，查看命中的切片与分数构成</div>
          <div className="mt-1 text-[12px] text-[var(--text-faint)]">这里测试的检索与智能体 <code className="font-mono">knowledge_search</code> 工具走同一路径</div>
        </div>
      ) : hits.length === 0 ? (
        <div className="grid place-items-center rounded-[var(--radius-lg)] border border-dashed border-[var(--border)] py-14 text-center">
          <Search className="h-6 w-6 text-[var(--text-faint)]" />
          <div className="mt-3 text-sm text-[var(--text-muted)]">无命中结果</div>
          <div className="mt-1 text-[12px] text-[var(--text-faint)]">尝试放宽阈值、切换检索模式，或确认文档已入库并索引</div>
        </div>
      ) : (
        <>
          <div className="text-[12.5px] text-[var(--text-muted)]">
            命中 <b className="text-[var(--text)]">{total}</b> 条 · 显示前 {hits.length} · 模式 {mode}
          </div>
          {hits.map((h, i) => <Hit key={h.chunk_id} hit={h} rank={i + 1} mode={mode} query={lastQuery} />)}
          {hasMore && (
            <div className="flex justify-center pt-1">
              <button className={BTN_GHOST} disabled={loadingMore} onClick={() => void loadMore()}>
                {loadingMore ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                加载更多（{hits.length}/{total}）
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Hit({ hit, rank, mode, query }: { hit: KnowledgeHit; rank: number; mode: string; query: string }) {
  const chunkIdx = hit.metadata?.chunk_index;
  return (
    <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-elevated)] p-3.5">
      <div className="flex items-center gap-2.5">
        <span className="w-6 font-mono text-[11px] text-[var(--text-faint)]">#{rank}</span>
        <span className="text-[13px] font-semibold">{hit.title}</span>
        <span className="truncate font-mono text-[11px] text-[var(--text-faint)]">
          {hit.source_uri}{typeof chunkIdx === "number" ? ` · chunk ${chunkIdx}` : ""}
        </span>
        <span className="ml-auto font-mono text-xs font-semibold text-[var(--accent)]">{hit.score.toFixed(3)}</span>
      </div>
      <p className="mt-2 text-[13px] leading-relaxed text-[var(--text-muted)] line-clamp-4">{highlight(hit.text, query)}</p>
      <div className="mt-2.5 grid max-w-[360px] grid-cols-[auto_1fr_auto] items-center gap-x-2 gap-y-1">
        <ScoreBar label="向量" value={hit.vector_score} color="var(--accent)" dim={mode === "keyword"} />
        <ScoreBar label="关键词" value={hit.keyword_score} color="#8b5cf6" dim={mode === "vector"} />
      </div>
    </div>
  );
}

function ScoreBar({ label, value, color, dim }: { label: string; value: number; color: string; dim?: boolean }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <>
      <span className={cn("text-[10.5px] uppercase text-[var(--text-faint)]", dim && "opacity-40")}>{label}</span>
      <span className={cn("h-[5px] overflow-hidden rounded-full bg-[var(--surface-3)]", dim && "opacity-40")}>
        <i className="block h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
      </span>
      <span className={cn("font-mono text-[10.5px] text-[var(--text-muted)] tabular-nums", dim && "opacity-40")}>{value.toFixed(2)}</span>
    </>
  );
}

// ---------- Create ---------- //
function CreateDrawer({ open, onClose, onCreated }: {
  open: boolean; onClose: () => void; onCreated: (id: string) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [visibility, setVisibility] = useState("private");
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(150);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (!name.trim()) return;
    setBusy(true);
    try {
      const kb = await createKnowledgeBase({
        name: name.trim(), description: description.trim(), visibility,
        default_parser_config: { chunk_size: chunkSize, chunk_overlap: chunkOverlap },
      });
      setName(""); setDescription(""); setVisibility("private");
      setChunkSize(1000); setChunkOverlap(150);
      toast.success("知识库已创建");
      onCreated(kb.id);
    } catch (e) { toast.error(e instanceof Error ? e.message : "创建失败"); }
    finally { setBusy(false); }
  };

  return (
    <Drawer open={open} title="新建知识库" onClose={onClose}
      footer={<>
        <button className={BTN_GHOST} onClick={onClose}>取消</button>
        <button className={BTN_PRIMARY} disabled={busy || !name.trim()} onClick={submit}>
          {busy && <Loader2 className="h-4 w-4 animate-spin" />} 创建
        </button>
      </>}>
      <Field label="名称"><input className={INPUT} value={name} onChange={(e) => setName(e.target.value)} placeholder="如：PAI 产品文档" /></Field>
      <Field label="描述"><input className={INPUT} value={description} onChange={(e) => setDescription(e.target.value)} placeholder="可选" /></Field>
      <Field label="可见性">
        <select className={INPUT} value={visibility} onChange={(e) => setVisibility(e.target.value)}>
          <option value="private">private — 仅自己</option>
          <option value="workspace">workspace — 团队可查询</option>
          <option value="public">public</option>
        </select>
      </Field>
      <div className="mt-1 grid grid-cols-2 gap-3">
        <Field label="chunk_size"><input type="number" className={cn(INPUT, "font-mono")} value={chunkSize} onChange={(e) => setChunkSize(Math.max(100, Number(e.target.value) || 0))} /></Field>
        <Field label="chunk_overlap"><input type="number" className={cn(INPUT, "font-mono")} value={chunkOverlap} onChange={(e) => setChunkOverlap(Math.max(0, Number(e.target.value) || 0))} /></Field>
      </div>
      <div className="flex items-start gap-2 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2.5 text-[12px] text-[var(--text-muted)]">
        <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
        <div>创建后使用内置 <b>local</b> embedding 与 local_sql 向量库。接入外部 Provider 后可在「配置」中切换。</div>
      </div>
    </Drawer>
  );
}

// --- utils ----------------------------------------------------------------- //
function fmtTime(iso?: string): string {
  if (!iso) return "—";
  const d = new Date(iso), diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return "刚刚";
  if (diff < 3600) return `${Math.floor(diff / 60)}分钟前`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}小时前`;
  return d.toLocaleDateString();
}

function highlight(text: string, query: string): ReactNode {
  const terms = query.toLowerCase().split(/\s+/).filter((t) => t.length > 1);
  if (!terms.length) return text;
  const re = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
  return text.split(re).map((p, i) =>
    terms.includes(p.toLowerCase())
      ? <mark key={i} className="rounded-[2px] bg-[var(--accent)]/25 px-0.5 text-[var(--text)]">{p}</mark>
      : <span key={i}>{p}</span>
  );
}
