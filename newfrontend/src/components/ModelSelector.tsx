const MODELS = ["gpt-4o-mini", "gpt-4o"];

export function ModelSelector({
  model,
  onChange,
}: {
  model: string;
  onChange: (m: string) => void;
}) {
  return (
    <select
      aria-label="Model"
      value={model}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-md border border-gray-300 px-2 py-1 text-sm"
    >
      {MODELS.map((m) => (
        <option key={m} value={m}>
          {m}
        </option>
      ))}
    </select>
  );
}
