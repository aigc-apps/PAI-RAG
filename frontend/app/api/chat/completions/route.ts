import { NextRequest, NextResponse } from 'next/server';
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8682"; // 你的后端地址

export async function POST(request: NextRequest) {
  const text = await request.text();

  const headers = new Headers(request.headers);
  // 删除 Next.js 自动添加的 header，避免冲突
  headers.delete('host');
  headers.delete('connection');
  // 删除 content-length，让 fetch 自动计算正确的长度
  headers.delete('content-length');

  // 创建 AbortController 用于超时控制（聊天接口可能需要更长时间）
  const timeoutMs = parseInt(process.env.CHAT_TIMEOUT_MS || '600000', 10); // 默认 10 分钟
  const controller = new AbortController();
  let timeoutId: NodeJS.Timeout | null = null;

  try {
    timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    const response = await fetch(`${BACKEND_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: headers,
      body: text,
      signal: controller.signal,
    });

    if (timeoutId) clearTimeout(timeoutId);

    // Check if the response is ok (e.g. 200)
    if (!response.ok) {
      return new NextResponse('Error calling chat api', { status: response.status });
    }

    // Create a ReadableStream to handle the streaming response
    const stream = new ReadableStream({
      async start(controller) {
        try {
          if (!response.body) {
            controller.error(new Error('Response body is null'));
            return;
          }
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
  } catch (error: any) {
    if (timeoutId) clearTimeout(timeoutId);
    console.log("Chat API request failed: ", error);
    
    // 处理超时错误
    if (error.name === 'AbortError' || error.code === 'UND_ERR_HEADERS_TIMEOUT') {
      return new NextResponse(
        JSON.stringify({ 
          error: 'Chat API request timeout', 
          message: `Request exceeded timeout of ${timeoutMs}ms` 
        }),
        { 
          status: 504,
          headers: { 'Content-Type': 'application/json' }
        }
      );
    }
    
    return new NextResponse(
      JSON.stringify({ 
        error: 'Chat API request failed', 
        message: error.message || String(error) 
      }),
      { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}
