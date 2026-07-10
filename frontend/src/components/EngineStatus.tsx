import { useCallback, useEffect, useState } from "react";
import { Database } from "lucide-react";
import { getSearchEngine, type SearchEngineStatus } from "../api/knowledge";
import { cn } from "../lib/cn";

/**
 * Shared retrieval-engine reachability UI. One source of truth for:
 * - deriving the reachability state from a live `SearchEngineStatus` (the ES
 *   ping via `GET /v1/knowledge/engine`), and
 * - rendering it, in the two shapes the app needs.
 *
 * Two call sites use this: the KB *browse* view (a status pill naming the active
 * engine) and the KB *settings* tab (a compact reachability dot). They share the
 * state logic and colour tokens here so the two never drift.
 */

export type ReachState = "unconfigured" | "checking" | "reachable" | "unreachable";

/** The single place that decides reachability from a live status + local hints. */
export function reachStateOf(
  engine: SearchEngineStatus | null,
  opts?: { configured?: boolean; checking?: boolean }
): ReachState {
  const configured = opts?.configured ?? !!engine?.configured;
  if (!configured) return "unconfigured";
  if (!engine) return opts?.checking ? "checking" : "unreachable";
  return engine.configured && engine.healthy ? "reachable" : "unreachable";
}

/**
 * Probe the live retrieval engine and expose `{ engine, checking, refresh }`.
 * `enabled` gates the request (skip it for the always-reachable local engine);
 * `key` (e.g. the ES URL) re-probes when it changes — a settings save rebuilds
 * the backend engine, making the previous probe stale.
 */
export function useEngineStatus(enabled: boolean = true, key?: string) {
  const [engine, setEngine] = useState<SearchEngineStatus | null>(null);
  const [checking, setChecking] = useState(false);
  const refresh = useCallback(async () => {
    setChecking(true);
    try {
      setEngine(await getSearchEngine());
    } catch {
      setEngine(null);
    } finally {
      setChecking(false);
    }
  }, []);
  useEffect(() => {
    if (!enabled) {
      setEngine(null);
      return;
    }
    void refresh();
  }, [enabled, key, refresh]);
  return { engine, checking, refresh };
}

const WARN = "text-[var(--warning,#d97706)]";

/**
 * The retrieval-engine reachability badge, in two variants:
 * - `"pill"` (browse view): a bordered pill naming the active engine, with a
 *   "· 不可达" suffix when a configured ES can't be reached. Renders nothing
 *   until the first probe resolves.
 * - `"dot"` (settings tab): a compact coloured dot + label
 *   (未配置 / 检测中… / 可达 / 不可达). Pass `configured`/`checking` so it can
 *   show state before/independently of the probe result.
 */
export function EngineStatusBadge({
  engine,
  variant = "pill",
  configured,
  checking = false,
}: {
  engine: SearchEngineStatus | null;
  variant?: "pill" | "dot";
  configured?: boolean;
  checking?: boolean;
}) {
  const state = reachStateOf(engine, { configured, checking });

  if (variant === "pill") {
    if (!engine) return null;
    const isEs = engine.engine === "elasticsearch";
    const ok = !engine.configured || engine.healthy;
    return (
      <span
        className="inline-flex items-center gap-1.5 rounded-full border border-[var(--border)] bg-[var(--surface)] px-2.5 py-1 text-[11.5px] text-[var(--text-muted)]"
        title={
          isEs
            ? engine.healthy
              ? engine.detail || "Elasticsearch 混合检索（BM25 + 向量 kNN）已连接"
              : engine.detail || "已配置 Elasticsearch 但当前不可达，自动降级本地检索"
            : "内置本地检索引擎（配置 ELASTICSEARCH_URL 可启用 Elasticsearch 混合检索）"
        }
      >
        <Database className={cn("h-3.5 w-3.5", ok ? "text-[var(--accent)]" : WARN)} />
        {isEs ? "Elasticsearch" : "本地检索"}
        {engine.configured && !engine.healthy && <span className={WARN}>· 不可达</span>}
      </span>
    );
  }

  // variant === "dot"
  if (state === "unconfigured") {
    return <span className="text-xs text-[var(--text-faint)]">未配置</span>;
  }
  if (state === "checking") {
    return <span className="text-xs text-[var(--text-faint)]">检测中…</span>;
  }
  const cls = "inline-flex items-center gap-1.5 text-xs";
  const dot = "h-1.5 w-1.5 rounded-full";
  if (state === "reachable") {
    return (
      <span className={cn(cls, "text-[var(--success)]")} title={engine?.detail || "已连接"}>
        <span className={cn(dot, "bg-[var(--success)]")} />
        可达
      </span>
    );
  }
  return (
    <span className={cn(cls, WARN)} title={engine?.detail || "无法连接到 Elasticsearch"}>
      <span className={cn(dot, "bg-[var(--warning,#d97706)]")} />
      不可达
    </span>
  );
}
