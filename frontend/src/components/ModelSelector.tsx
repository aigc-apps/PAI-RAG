import { useEffect, useRef, useState } from "react";
import { listModels } from "../api/models";
import { useI18n } from "../i18n";

export function ModelSelector({
  model,
  onChange,
}: {
  model: string;
  onChange: (m: string) => void;
}) {
  const { t } = useI18n();
  const [ids, setIds] = useState<string[]>([]);
  const [defaultId, setDefaultId] = useState("");
  const modelRef = useRef(model);
  modelRef.current = model;

  useEffect(() => {
    let cancelled = false;
    listModels()
      .then(({ ids: fetched, default: def }) => {
        if (cancelled) return;
        setIds(fetched);
        setDefaultId(def);
        const current = modelRef.current;
        const valid = current && fetched.includes(current);
        if (!valid && def) onChange(def);
      })
      .catch(() => {
        /* keep the fallback (current model) on error */
      });
    return () => {
      cancelled = true;
    };
  }, [onChange]);

  const display = model || defaultId;
  const options = ids.includes(display) ? ids : display ? [display, ...ids] : ids;

  return (
    <select
      aria-label={t("model.aria")}
      value={display}
      onChange={(e) => onChange(e.target.value)}
      className="text-xs font-medium rounded-[var(--radius-sm)] px-2 py-1 text-[var(--text-muted)] bg-transparent hover:bg-[var(--surface-2)] hover:text-[var(--text)] border border-transparent focus:border-[var(--border-strong)] outline-none cursor-pointer transition-colors"
    >
      {options.map((m) => (
        <option key={m} value={m}>
          {m}
        </option>
      ))}
    </select>
  );
}
