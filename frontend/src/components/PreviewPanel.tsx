import { useEffect, useState } from "react";
import { Download, Loader2, Maximize2, Minimize2, X } from "lucide-react";
import type { FileArtifact } from "../types";
import { cn } from "../lib/cn";
import { fileUrl, humanSize } from "../lib/files";
import { usePreviewStore } from "../store/preview";
import { ArtifactIcon } from "./ArtifactIcon";
import { Markdown } from "./Markdown";

/** Fetches an artifact's text body (for markdown / text kinds). */
function useTextBody(artifact: FileArtifact, enabled: boolean) {
  const [state, setState] = useState<{
    loading: boolean;
    text: string;
    error: string | null;
  }>({ loading: enabled, text: "", error: null });

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    setState({ loading: true, text: "", error: null });
    fetch(fileUrl(artifact.id))
      .then(async (r) => {
        if (!r.ok) throw new Error(`加载失败 (${r.status})`);
        return r.text();
      })
      .then((text) => {
        if (!cancelled) setState({ loading: false, text, error: null });
      })
      .catch((err: unknown) => {
        if (!cancelled)
          setState({
            loading: false,
            text: "",
            error: err instanceof Error ? err.message : "加载失败",
          });
      });
    return () => {
      cancelled = true;
    };
  }, [artifact.id, enabled]);

  return state;
}

function PreviewBody({ artifact }: { artifact: FileArtifact }) {
  const isText = artifact.kind === "markdown" || artifact.kind === "text";
  const { loading, text, error } = useTextBody(artifact, isText);
  const url = fileUrl(artifact.id);

  if (artifact.kind === "image") {
    return (
      <div className="grid h-full place-items-center p-4">
        <img
          src={url}
          alt={artifact.name}
          className="max-h-full max-w-full rounded-[var(--radius-md)] object-contain"
        />
      </div>
    );
  }

  if (artifact.kind === "html") {
    return (
      <iframe
        title={artifact.name}
        src={url}
        // Sandboxed: allow the document to run its own scripts (charts, export
        // buttons) and trigger downloads (jsPDF etc.), but withhold
        // allow-same-origin so it cannot read the app's cookies/storage.
        sandbox="allow-scripts allow-downloads"
        className="h-full w-full border-0 bg-white"
      />
    );
  }

  if (artifact.kind === "file") {
    return (
      <div className="grid h-full place-items-center p-6 text-center text-[var(--text-muted)]">
        <div>
          <p className="mb-3 text-sm">此文件类型不支持预览。</p>
          <a
            href={url}
            download={artifact.name}
            className="inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] border border-[var(--border)] px-3 py-1.5 text-sm text-[var(--text)] hover:bg-[var(--surface-2)]"
          >
            <Download className="h-3.5 w-3.5" /> 下载 {artifact.name}
          </a>
        </div>
      </div>
    );
  }

  // markdown / text
  if (loading)
    return (
      <div className="grid h-full place-items-center text-[var(--text-muted)]">
        <Loader2 className="h-5 w-5 animate-spin" />
      </div>
    );
  if (error)
    return (
      <div className="grid h-full place-items-center p-6 text-center text-sm text-[var(--danger)]">
        {error}
      </div>
    );
  if (artifact.kind === "markdown")
    return (
      <div className="overflow-auto p-5">
        <Markdown content={text} />
      </div>
    );
  return (
    <pre className="h-full overflow-auto whitespace-pre-wrap break-words p-4 font-mono text-xs leading-6 text-[var(--text)]">
      {text}
    </pre>
  );
}

/** Horizontal, scrollable strip to switch between a message's artifacts.
 * Rendered only when there is more than one. */
function ArtifactSwitcher({
  items,
  activeId,
  onSelect,
}: {
  items: FileArtifact[];
  activeId: string;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="flex shrink-0 gap-1 overflow-x-auto border-b border-[var(--border)] px-2 py-1.5">
      {items.map((f) => {
        const active = f.id === activeId;
        return (
          <button
            key={f.id}
            type="button"
            onClick={() => onSelect(f.id)}
            title={f.name}
            className={cn(
              "inline-flex items-center gap-1.5 whitespace-nowrap rounded-[var(--radius-sm)] px-2 py-1 text-xs transition-colors",
              active
                ? "bg-[var(--surface-3)] text-[var(--text)]"
                : "text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
            )}
          >
            <ArtifactIcon kind={f.kind} className="h-3.5 w-3.5 shrink-0" />
            <span className="max-w-[140px] truncate">{f.name}</span>
          </button>
        );
      })}
    </div>
  );
}

export function PreviewPanel() {
  const items = usePreviewStore((s) => s.items);
  const activeId = usePreviewStore((s) => s.activeId);
  const setActive = usePreviewStore((s) => s.setActive);
  const expanded = usePreviewStore((s) => s.expanded);
  const toggleExpanded = usePreviewStore((s) => s.toggleExpanded);
  const close = usePreviewStore((s) => s.close);
  const artifact = items.find((i) => i.id === activeId) ?? items[0];
  if (!artifact) return null;

  return (
    <aside
      className={cn(
        "flex flex-col border-l border-[var(--border)] bg-[var(--surface)]",
        expanded ? "flex-1 min-w-0" : "w-[42%] min-w-[320px] max-w-[720px]"
      )}
    >
      <header className="flex h-12 shrink-0 items-center gap-2 border-b border-[var(--border)] px-3">
        <span className="truncate text-sm font-medium text-[var(--text)]" title={artifact.name}>
          {artifact.name}
        </span>
        {artifact.size >= 0 && (
          <span className="shrink-0 text-xs text-[var(--text-faint)]">
            {humanSize(artifact.size)}
          </span>
        )}
        <div className="ml-auto flex items-center gap-1">
          <a
            href={fileUrl(artifact.id)}
            download={artifact.name}
            aria-label="下载"
            className="rounded-[var(--radius-sm)] p-1.5 text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
          >
            <Download className="h-4 w-4" />
          </a>
          <button
            type="button"
            aria-label={expanded ? "还原" : "最大化"}
            onClick={toggleExpanded}
            className="rounded-[var(--radius-sm)] p-1.5 text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
          >
            {expanded ? (
              <Minimize2 className="h-4 w-4" />
            ) : (
              <Maximize2 className="h-4 w-4" />
            )}
          </button>
          <button
            type="button"
            aria-label="关闭预览"
            onClick={close}
            className="rounded-[var(--radius-sm)] p-1.5 text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </header>
      {items.length > 1 && (
        <ArtifactSwitcher items={items} activeId={artifact.id} onSelect={setActive} />
      )}
      <div className="min-h-0 flex-1 overflow-hidden">
        {/* key on id so per-artifact fetch/state resets cleanly on switch */}
        <PreviewBody key={artifact.id} artifact={artifact} />
      </div>
    </aside>
  );
}
