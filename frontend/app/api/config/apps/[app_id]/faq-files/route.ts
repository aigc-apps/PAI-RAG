// app/api/proxy/route.js
import { NextRequest } from 'next/server';
import { proxyRequest } from '@/app/api/proxy';

export async function POST(request: NextRequest) {
  return proxyRequest(request);
}

