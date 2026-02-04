// Load prompts based on environment
let prompts: {
  react_prompt: string;
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
export const REACT_PROMPT = prompts?.react_prompt || '';

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
