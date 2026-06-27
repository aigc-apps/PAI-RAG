export async function listModels(): Promise<string[]> {
  const res = await fetch("/v1/models");
  if (!res.ok) throw new Error(`models request failed: ${res.status}`);
  const body = (await res.json()) as { data: { id: string }[] };
  return body.data.map((m) => m.id);
}
