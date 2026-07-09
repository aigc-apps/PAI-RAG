import { apiFetch } from "../lib/apiFetch";

async function parseJson<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const message = body?.error?.message || body?.detail || `request failed: ${res.status}`;
    throw new Error(message);
  }
  return body as T;
}

// --- config shapes (opaque JSON columns on the KB row) --------------------- //
export interface EmbeddingConfig {
  provider_id: string;
  model: string;
  dimension: number;
  normalize: boolean;
}
export interface VectorStoreConfig {
  provider_id: string;
  index_name: string;
  namespace: string;
  metric: string;
  dimension: number;
}
export interface RerankConfig {
  enabled: boolean;
  provider_id?: string;
  model?: string;
  top_n?: number;
}
export interface KeywordIndexConfig {
  provider_id: string;
  enabled: boolean;
}
export interface ParserConfig {
  chunk_size: number;
  chunk_overlap: number;
}
export interface RetrievalConfig {
  mode: "hybrid" | "vector" | "keyword";
  top_k: number;
  score_threshold: number;
  force_citation: boolean;
}

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string;
  owner_user_id: string;
  visibility: string;
  status: string;
  document_count: number;
  chunk_count: number;
  active_index_version_id?: string | null;
  default_parser_config: Partial<ParserConfig>;
  default_retrieval_config: Partial<RetrievalConfig>;
  embedding_config: Partial<EmbeddingConfig>;
  vector_store_config: Partial<VectorStoreConfig>;
  keyword_index_config: Partial<KeywordIndexConfig>;
  rerank_config: Partial<RerankConfig>;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeDocument {
  id: string;
  kb_id: string;
  uri: string;
  source_type: string;
  title: string;
  description: string;
  status: string;
  tags: string[];
  category?: string;
  chunk_count: number;
  indexed_at?: string;
}

export interface KnowledgeChunk {
  id: string;
  kb_id: string;
  document_id: string;
  chunk_index: number;
  text: string;
  token_count: number;
  status: string;
}

export interface KnowledgeDataSourceReport {
  discovered?: number;
  added?: number;
  updated?: number;
  unchanged?: number;
  deleted?: number;
  failed?: number;
  errors?: { path?: string; error?: string }[];
}

export type DataSourceStatus = "idle" | "syncing" | "succeeded" | "partial" | "failed";

export interface KnowledgeDataSource {
  id: string;
  kb_id: string;
  name: string;
  source_key: string;
  source_type: string;
  source_config: Record<string, unknown>;
  enabled: boolean;
  status: DataSourceStatus;
  doc_count: number;
  last_sync_at?: string | null;
  last_sync_finished_at?: string | null;
  last_error?: string | null;
  last_sync_report: KnowledgeDataSourceReport;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeHit {
  kb_id: string;
  document_id: string;
  chunk_id: string;
  title: string;
  source_uri: string;
  source_type: string;
  text: string;
  score: number;
  vector_score: number;
  keyword_score: number;
  metadata: Record<string, unknown>;
}

// Paginated list envelope returned by documents / chunks / search.
export interface Paginated<T> {
  data: T[];
  total: number;
  offset: number;
  limit: number;
  has_more: boolean;
}

export interface SearchEngineStatus {
  engine: "elasticsearch" | "local" | string;
  configured: boolean;
  healthy: boolean;
}

export async function listKnowledgeBases(): Promise<KnowledgeBase[]> {
  const body = await parseJson<{ data: KnowledgeBase[] }>(
    await apiFetch("/v1/knowledge-bases")
  );
  return body.data;
}

export async function createKnowledgeBase(payload: {
  name: string;
  description?: string;
  visibility?: string;
  default_parser_config?: Partial<ParserConfig>;
  default_retrieval_config?: Partial<RetrievalConfig>;
  // Chosen models by qualified id ("provider/model"). embedding is frozen at
  // creation (omit → inherit the catalog default); rerank is optional.
  embedding_model?: string;
  rerank_model?: string;
  rerank_top_n?: number;
}): Promise<KnowledgeBase> {
  return parseJson<KnowledgeBase>(
    await apiFetch("/v1/knowledge-bases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

export type KnowledgeBasePatch = Partial<{
  name: string;
  description: string;
  visibility: string;
  default_parser_config: Partial<ParserConfig>;
  default_retrieval_config: Partial<RetrievalConfig>;
  // embedding is immutable after creation. rerank is mutable via friendly fields.
  rerank_model: string;
  rerank_enabled: boolean;
  rerank_top_n: number;
}>;

export async function updateKnowledgeBase(
  kbId: string,
  patch: KnowledgeBasePatch
): Promise<KnowledgeBase> {
  return parseJson<KnowledgeBase>(
    await apiFetch(`/v1/knowledge-bases/${kbId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    })
  );
}

export async function deleteKnowledgeBase(kbId: string): Promise<void> {
  await parseJson(
    await apiFetch(`/v1/knowledge-bases/${kbId}`, { method: "DELETE" })
  );
}

export async function importKnowledgeDocument(
  kbId: string,
  payload: {
    title: string;
    content: string;
    uri?: string;
    source_type?: string;
    tags?: string[];
    category?: string;
  }
): Promise<{ document: KnowledgeDocument; job_id: string }> {
  // 202: the document comes back as `processing`; the background worker ingests
  // and flips it to `indexed`. Poll the document list to observe the transition.
  return parseJson<{ document: KnowledgeDocument; job_id: string }>(
    await apiFetch(`/v1/knowledge-bases/${kbId}/documents/import`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

export async function uploadKnowledgeDocument(
  kbId: string,
  file: File,
  meta?: { title?: string; tags?: string[]; category?: string }
): Promise<{ document: KnowledgeDocument; job_id: string }> {
  const form = new FormData();
  form.append("file", file);
  if (meta?.title) form.append("title", meta.title);
  if (meta?.tags && meta.tags.length) form.append("tags", meta.tags.join(","));
  if (meta?.category) form.append("category", meta.category);
  // No Content-Type header — the browser sets multipart/form-data + boundary.
  // 202: document returns `processing`; worker ingests → `indexed` (poll to observe).
  return parseJson<{ document: KnowledgeDocument; job_id: string }>(
    await apiFetch(`/v1/knowledge-bases/${kbId}/documents/upload`, {
      method: "POST",
      body: form,
    })
  );
}

export async function getUploadSupport(): Promise<{ extensions: string[]; max_mb: number }> {
  return parseJson<{ extensions: string[]; max_mb: number }>(
    await apiFetch("/v1/knowledge/upload-support")
  );
}

export async function listKnowledgeDocuments(
  kbId: string,
  filters?: {
    source_type?: string; status?: string; category?: string; tag?: string; query?: string;
    limit?: number; offset?: number;
  }
): Promise<Paginated<KnowledgeDocument>> {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(filters || {})) {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  }
  const suffix = qs.toString() ? `?${qs.toString()}` : "";
  return parseJson<Paginated<KnowledgeDocument>>(
    await apiFetch(`/v1/knowledge-bases/${kbId}/documents${suffix}`)
  );
}

export async function listKnowledgeChunks(
  kbId: string,
  documentId?: string,
  opts?: { limit?: number; offset?: number }
): Promise<Paginated<KnowledgeChunk>> {
  const qs = new URLSearchParams();
  if (documentId) qs.set("document_id", documentId);
  if (opts?.limit !== undefined) qs.set("limit", String(opts.limit));
  if (opts?.offset !== undefined) qs.set("offset", String(opts.offset));
  const suffix = qs.toString() ? `?${qs.toString()}` : "";
  return parseJson<Paginated<KnowledgeChunk>>(
    await apiFetch(`/v1/knowledge-bases/${kbId}/chunks${suffix}`)
  );
}

export async function searchKnowledge(payload: {
  kb_ids: string[];
  query: string;
  mode?: "hybrid" | "vector" | "keyword";
  top_k?: number;
  offset?: number;
  score_threshold?: number;
  filters?: Record<string, unknown>;
}): Promise<Paginated<KnowledgeHit>> {
  return parseJson<Paginated<KnowledgeHit>>(
    await apiFetch("/v1/knowledge/query/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

export async function getSearchEngine(): Promise<SearchEngineStatus> {
  return parseJson<SearchEngineStatus>(await apiFetch("/v1/knowledge/engine"));
}

// --- data sources ---------------------------------------------------------- //

export async function listDataSources(kbId: string): Promise<KnowledgeDataSource[]> {
  const body = await parseJson<{ data: KnowledgeDataSource[] }>(
    await apiFetch(`/v1/knowledge-bases/${kbId}/datasources`)
  );
  return body.data;
}

export async function getDataSource(
  kbId: string,
  dsId: string
): Promise<KnowledgeDataSource> {
  return parseJson<KnowledgeDataSource>(
    await apiFetch(`/v1/knowledge-bases/${kbId}/datasources/${dsId}`)
  );
}

export async function createDataSource(
  kbId: string,
  payload: {
    name: string;
    source_type: string;
    source_config: Record<string, unknown>;
    enabled?: boolean;
  }
): Promise<KnowledgeDataSource> {
  return parseJson<KnowledgeDataSource>(
    await apiFetch(`/v1/knowledge-bases/${kbId}/datasources`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

export type DataSourcePatch = Partial<{
  name: string;
  source_config: Record<string, unknown>;
  enabled: boolean;
}>;

export async function updateDataSource(
  kbId: string,
  dsId: string,
  patch: DataSourcePatch
): Promise<KnowledgeDataSource> {
  return parseJson<KnowledgeDataSource>(
    await apiFetch(`/v1/knowledge-bases/${kbId}/datasources/${dsId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    })
  );
}

export async function deleteDataSource(kbId: string, dsId: string): Promise<void> {
  await parseJson(
    await apiFetch(`/v1/knowledge-bases/${kbId}/datasources/${dsId}`, {
      method: "DELETE",
    })
  );
}

export async function syncDataSource(
  kbId: string,
  dsId: string
): Promise<{ ok: boolean; status: string; data_source: KnowledgeDataSource }> {
  return parseJson<{ ok: boolean; status: string; data_source: KnowledgeDataSource }>(
    await apiFetch(`/v1/knowledge-bases/${kbId}/datasources/${dsId}/sync`, {
      method: "POST",
    })
  );
}
