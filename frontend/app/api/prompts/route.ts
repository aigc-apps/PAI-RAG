import { NextResponse } from 'next/server';
import { readFile } from 'fs/promises';
import { resolve } from 'path';
import yaml from 'js-yaml';

/**
 * API route to serve prompts from YAML file
 * Route: /api/prompts
 */
export async function GET() {
  try {
    // Get the project root directory
    // In Next.js, process.cwd() is the project root (/mnt/ranxia/PAI-RAG/frontend)
    // So we need to go up one level to get to the project root
    const projectRoot = resolve(process.cwd(), '..');
    const promptsFile = resolve(projectRoot, 'resources/prompts/prompts.yaml');

    // Read YAML file
    const fileContent = await readFile(promptsFile, 'utf-8');
    const prompts = yaml.load(fileContent) as Record<string, string>;

    return NextResponse.json({
      data: {
        react_prompt: prompts.react_prompt  || '',
      },
    });
  } catch (error: any) {
    console.error('Failed to load prompts:', error);
    return NextResponse.json(
      { error: 'Failed to load prompts', message: error.message },
      { status: 500 }
    );
  }
}

