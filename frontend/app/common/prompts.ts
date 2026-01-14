// Load prompts based on environment
let prompts: {
  plan_prompt: string;
  act_prompt: string;
  act_with_plan_prompt: string;
  summary_prompt: string;
} | null = null;

// Only load prompts on server-side
if (typeof window === 'undefined') {
  try {
    // Dynamic import to avoid bundling server-only code in client
    const { loadPromptsFromFile } = require('./prompts.server');
    prompts = loadPromptsFromFile();
  } catch (error) {
    console.error('Failed to load prompts on server-side:', error);
    // Set to null to indicate failure
    prompts = null;
  }
}

// Export prompts
// On server-side: loaded from YAML file (or null if failed)
// On client-side: empty strings (components should use getPrompts() or fetch from API)
export const PLAN_PROMPT = prompts?.plan_prompt || '';
export const ACT_PROMPT = prompts?.act_prompt || '';
export const ACT_WITH_PLAN_PROMPT = prompts?.act_with_plan_prompt || '';
export const SUMMARY_PROMPT = prompts?.summary_prompt || '';

// Export function to get prompts (for async usage, e.g., in client components)
export async function getPrompts() {
  if (typeof window === 'undefined') {
    // Server-side: load from file
    const { loadPromptsFromFile } = require('./prompts.server');
    return loadPromptsFromFile();
  } else {
    // Client-side: fetch from API
    const response = await fetch('/api/prompts');
    if (!response.ok) {
      throw new Error(`Failed to fetch prompts: ${response.statusText}`);
    }
    const data = await response.json();
    return data.data;
  }
}
