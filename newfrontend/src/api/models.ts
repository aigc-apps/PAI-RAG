export interface ModelsCatalog {
  ids: string[];
  default: string;
}

export async function listModels(): Promise<ModelsCatalog> {
  const res = await fetch("/v1/models");
  if (!res.ok) throw new Error(`models request failed: ${res.status}`);
  const body = (await res.json()) as { data: { id: string }[]; default?: string };
  const ids = body.data.map((m) => m.id);
  return { ids, default: body.default ?? ids[0] ?? "" };
}
