import { FileText, FileCode, Image as ImageIcon, File as FileIcon } from "lucide-react";
import type { FileArtifact } from "../types";

/** Icon for a file artifact, chosen by its `kind`. Shared by the message-level
 * artifact strip and the preview panel's switcher. */
export function ArtifactIcon({
  kind,
  className = "h-3.5 w-3.5 shrink-0",
}: {
  kind: FileArtifact["kind"];
  className?: string;
}) {
  if (kind === "image") return <ImageIcon className={className} />;
  if (kind === "html") return <FileCode className={className} />;
  if (kind === "markdown" || kind === "text") return <FileText className={className} />;
  return <FileIcon className={className} />;
}
