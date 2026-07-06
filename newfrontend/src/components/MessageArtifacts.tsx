import type { FileArtifact } from "../types";
import { humanSize } from "../lib/files";
import { usePreviewStore } from "../store/preview";
import { ArtifactIcon } from "./ArtifactIcon";

/** Files an assistant message produced, surfaced at the message level (not
 * buried inside a tool-call record). Clicking a card opens the preview panel
 * on the whole set, focused on that file — so the panel can switch between
 * siblings. Download-only kinds ("file") open the panel's download view. */
export function MessageArtifacts({ files }: { files: FileArtifact[] }) {
  const open = usePreviewStore((s) => s.open);
  const unique = files.filter((f, i) => files.findIndex((x) => x.id === f.id) === i);
  if (unique.length === 0) return null;

  return (
    <div className="mt-2.5 flex flex-wrap gap-2">
      {unique.map((f) => (
        <button
          key={f.id}
          type="button"
          onClick={() => open(unique, f.id)}
          className="group flex items-center gap-2.5 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-left transition-colors hover:border-[var(--accent)]/50 hover:bg-[var(--surface-2)]"
        >
          <span className="grid h-8 w-8 shrink-0 place-items-center rounded-[var(--radius-sm)] bg-[var(--surface-3)] text-[var(--text-muted)] group-hover:text-[var(--accent)]">
            <ArtifactIcon kind={f.kind} className="h-4 w-4" />
          </span>
          <span className="min-w-0">
            <span className="block max-w-[220px] truncate text-sm font-medium text-[var(--text)]">
              {f.name}
            </span>
            <span className="block text-xs text-[var(--text-faint)]">
              {f.kind === "file" ? "下载" : "预览"}
              {f.size >= 0 ? ` · ${humanSize(f.size)}` : ""}
            </span>
          </span>
        </button>
      ))}
    </div>
  );
}
