import { useEffect, useState } from "react";
import { listModels } from "../api/models";

export function ModelSelector({
  model,
  onChange,
}: {
  model: string;
  onChange: (m: string) => void;
}) {
  const [models, setModels] = useState<string[]>([model]);

  useEffect(() => {
    let cancelled = false;
    listModels()
      .then((ids) => {
        if (!cancelled && ids.length) setModels(ids);
      })
      .catch(() => {
        /* keep the fallback (current model) on error */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Always include the current model so the controlled <select> has a valid option.
  const options = models.includes(model) ? models : [model, ...models];

  return (
    <select
      aria-label="Model"
      value={model}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-md border border-gray-300 px-2 py-1 text-sm"
    >
      {options.map((m) => (
        <option key={m} value={m}>
          {m}
        </option>
      ))}
    </select>
  );
}
