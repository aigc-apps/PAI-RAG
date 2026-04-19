import { NextRequest } from 'next/server';
import { proxyRequest } from '@/app/api/proxy';

export async function GET(request: NextRequest) {
  return proxyRequest(request);
}

export async function DELETE(request: NextRequest) {
  return proxyRequest(request);
}
