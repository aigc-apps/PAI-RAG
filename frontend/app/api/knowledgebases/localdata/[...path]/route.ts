import { NextRequest, NextResponse } from 'next/server';
import { readFile, stat } from 'fs/promises';
import { resolve, normalize } from 'path';

// In Next.js, process.cwd() is the project root (/root/PAI-RAG)
// In production, it might be at /app/localdata (symbolic link)
const LOCALDATA_BASE = resolve(process.cwd(), '../localdata');

/**
 * Serve static files from localdata directory
 * Route: /api/knowledgebases/localdata/** -> ./localdata/**
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    // Extract path from params
    // For [...path], params.path is an array of path segments
    const paramsData = await params;
    const filePath = paramsData.path ? paramsData.path.join('/') : '';

    // Security: Prevent path traversal attacks
    const normalizedPath = normalize(filePath);
    if (normalizedPath.includes('..') || normalizedPath.startsWith('/')) {
      return NextResponse.json(
        { error: 'Invalid path' },
        { status: 400 }
      );
    }

    // Resolve full file path
    const fullPath = resolve(LOCALDATA_BASE, normalizedPath);
    console.log('fullPath', fullPath);

    // Security: Ensure the resolved path is within LOCALDATA_BASE
    if (!fullPath.startsWith(resolve(LOCALDATA_BASE))) {
      return NextResponse.json(
        { error: 'Access denied' },
        { status: 403 }
      );
    }

    // Check if file exists
    try {
      const fileStat = await stat(fullPath);
      
      // If it's a directory, return 403 (we only serve files)
      if (fileStat.isDirectory()) {
        return NextResponse.json(
          { error: 'Directory listing not allowed' },
          { status: 403 }
        );
      }
    } catch (error: any) {
      if (error.code === 'ENOENT') {
        return NextResponse.json(
          { error: 'File not found' },
          { status: 404 }
        );
      }
      throw error;
    }

    // Read file content
    const fileContent = await readFile(fullPath);
    
    // Determine content type based on file extension
    const contentType = getContentType(fullPath);
    
    // Return file with appropriate headers
    // Convert Buffer to Uint8Array for NextResponse compatibility
    return new NextResponse(new Uint8Array(fileContent), {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=3600',
      },
    });
  } catch (error) {
    console.error('Error serving localdata file:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

/**
 * Determine content type based on file extension
 */
function getContentType(filePath: string): string {
  const ext = filePath.split('.').pop()?.toLowerCase();
  
  const contentTypes: Record<string, string> = {
    // Text
    'txt': 'text/plain',
    'md': 'text/markdown',
    'json': 'application/json',
    'jsonl': 'application/jsonl',
    'csv': 'text/csv',
    'html': 'text/html',
    'xml': 'application/xml',
    
    // Images
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'png': 'image/png',
    'gif': 'image/gif',
    'svg': 'image/svg+xml',
    'webp': 'image/webp',
    
    // Documents
    'pdf': 'application/pdf',
    'doc': 'application/msword',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'xls': 'application/vnd.ms-excel',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    
    // Archives
    'zip': 'application/zip',
    'tar': 'application/x-tar',
    'gz': 'application/gzip',
    
    // Audio/Video
    'mp3': 'audio/mpeg',
    'mp4': 'video/mp4',
    'wav': 'audio/wav',
  };
  
  return contentTypes[ext || ''] || 'application/octet-stream';
}
