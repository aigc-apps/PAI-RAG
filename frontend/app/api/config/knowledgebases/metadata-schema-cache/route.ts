import { NextRequest } from 'next/server';
import { proxyRequest } from '@/app/api/proxy';

export async function DELETE(request: NextRequest) {
  return proxyRequest(request);
}
