export interface ModelConfigurationParams {
  name: string;
  label: string;
}

export const OPENAI_MODELS: ModelConfigurationParams[] = [
  {
    name: "gpt-4o",
    label: "GPT-4o",
  },
  {
    name: "gpt-4o-mini",
    label: "GPT-4o mini",
  },
];

export const QWEN_MODELS: ModelConfigurationParams[] = [
  {
    name: "qwen-max",
    label: "Qwen-Max",
  },
  {
    name: "qwen-turbo",
    label: "Qwen-Turbo",
  },
];
/**
 * Ollama model names _MUST_ be prefixed with `"ollama-"`
 */

export const ALL_MODELS: ModelConfigurationParams[] = [
  ...OPENAI_MODELS,
  ...QWEN_MODELS,
];

export type OPENAI_MODEL_NAMES = (typeof OPENAI_MODELS)[number]["name"];
export type QWEN_MODEL_NAMES = (typeof QWEN_MODELS)[number]["name"];
export type ALL_MODEL_NAMES = OPENAI_MODEL_NAMES | QWEN_MODEL_NAMES;

export const MODEL_NAME_PROVIDER_MAP: Record<ALL_MODEL_NAMES, string> = {
  // OpenAI models
  "gpt-4o": "openai",
  "gpt-4o-mini": "openai",
  // qwen models
  "qwen-max": "qwen",
  "qwen-turbo": "qwen",
};
