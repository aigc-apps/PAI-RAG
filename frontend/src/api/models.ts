export type ModelType = "chat" | "embedding" | "rerank";

export interface ModelInfo {
  id: string;
  type: ModelType;
  dimension: number | null;
}

export interface ModelsCatalog {
  ids: string[];
  default: string;
  models: ModelInfo[];
  defaultEmbedding: string | null;
  defaultRerank: string | null;
}

interface ModelsResponse {
  data: { id: string; type?: ModelType; dimension?: number | null }[];
  default?: string;
  default_embedding?: string | null;
  default_rerank?: string | null;
}

export async function listModels(): Promise<ModelsCatalog> {
  const res = await fetch("/v1/models");
  if (!res.ok) throw new Error(`models request failed: ${res.status}`);
  const body = (await res.json()) as ModelsResponse;
  const models: ModelInfo[] = body.data.map((m) => ({
    id: m.id,
    type: m.type ?? "chat",
    dimension: m.dimension ?? null,
  }));
  const ids = models.map((m) => m.id);
  return {
    ids,
    default: body.default ?? ids[0] ?? "",
    models,
    defaultEmbedding: body.default_embedding ?? null,
    defaultRerank: body.default_rerank ?? null,
  };
}

/** Models of a given type, in catalog order. */
export function modelsByType(catalog: ModelsCatalog, type: ModelType): ModelInfo[] {
  return catalog.models.filter((m) => m.type === type);
}
