// This file is server-only and should never be imported in client components
import { readFileSync } from 'fs';
import { resolve } from 'path';
import yaml from 'js-yaml';

// Cache for prompts loaded from YAML
let _promptsCache: {
  plan_prompt: string;
  act_prompt: string;
  act_with_plan_prompt: string;
  summary_prompt: string;
} | null = null;

/**
 * Load prompts from YAML file (server-side only, synchronous)
 */
export function loadPromptsFromFile(): {
  plan_prompt: string;
  act_prompt: string;
  act_with_plan_prompt: string;
  summary_prompt: string;
} {
  if (_promptsCache) {
    return _promptsCache;
  }

  // Get the project root directory
  // In Next.js server-side, process.cwd() is the project root (/mnt/ranxia/PAI-RAG/frontend)
  const projectRoot = resolve(process.cwd(), '..');
  const promptsFile = resolve(projectRoot, 'resources/prompts/prompts.yaml');

  try {
    const fileContent = readFileSync(promptsFile, 'utf-8');
    const prompts = yaml.load(fileContent) as Record<string, string>;

    if (!prompts) {
      throw new Error(`Prompts file is empty or invalid: ${promptsFile}`);
    }

    const requiredKeys = ['plan_prompt', 'act_prompt', 'act_with_plan_prompt', 'summary_prompt'];
    const missingKeys = requiredKeys.filter(
      (key) => !prompts[key] || typeof prompts[key] !== 'string'
    );

    if (missingKeys.length > 0) {
      throw new Error(
        `Missing required prompt keys in YAML file: ${missingKeys.join(', ')}`
      );
    }

    _promptsCache = {
      plan_prompt: prompts.plan_prompt,
      act_prompt: prompts.act_prompt,
      act_with_plan_prompt: prompts.act_with_plan_prompt,
      summary_prompt: prompts.summary_prompt,
    };

    return _promptsCache;
  } catch (error: any) {
    const errorMsg = `Failed to load prompts from YAML file ${promptsFile}: ${error.message}`;
    console.error(errorMsg);
    throw new Error(errorMsg);
  }
}

