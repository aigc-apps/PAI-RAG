import { NextRequest, NextResponse } from 'next/server';
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8682"; // 你的后端地址

export async function POST(request: NextRequest) {
  const text = await request.text();

  const response = await fetch(`${BACKEND_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: new Headers(request.headers),
    body: text,
  });

  // Check if the response is ok (e.g. 200)
  if (!response.ok) {
    return new NextResponse('Error calling chat api', { status: response.status });
  }

  // Create a ReadableStream to handle the streaming response
  const stream = new ReadableStream({
    async start(controller) {
      try {
        for await (const chunk of response.body as any) {
          controller.enqueue(chunk);
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });

  return new NextResponse(stream, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'Transfer-Encoding': 'chunked',
    },
  });
}
