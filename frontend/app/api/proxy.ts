// app/api/proxy/route.js
import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8682"; // 你的后端地址

export async function proxyRequest(request: NextRequest) {
  const { pathname, searchParams } = new URL(request.url);
  const path = pathname?.replace(/^\/api\b/, '/v1');

  // 3. Build the final upstream URL
  const upstreamUrl = new URL(path, BACKEND_URL);
  // 4. Copy all original search params (except maybe 'path')
  for (const [key, value] of searchParams.entries()) {
    upstreamUrl.searchParams.append(key, value);
  }

  const method = request.method;
  let headers = new Headers(request.headers);
  // 删除 Next.js 自动添加的 header，避免冲突
  headers.delete('host');
  headers.delete('connection');
  headers.delete('content-length');

  let body;
  const contentType = headers.get('content-type');

  if (method === 'GET' || method === 'HEAD' || method === 'OPTIONS' || method === 'DELETE') {
    body = undefined;
  } else {
    // 处理 multipart/form-data 或普通 body
    // 注意：request.body 是 ReadableStream，只能读一次
    if (contentType && contentType.includes('multipart/form-data')) {
        // form-data：直接使用原始 body 流
        const formData = await request.formData();

        // 2. Create a new FormData to send to external API
        const externalFormData = new FormData();

        // 3. Copy all fields from incoming formData
        for (const [key, value] of formData.entries()) {
          if ( value && typeof value === 'object' && typeof value.arrayBuffer === 'function' && typeof value.type === 'string' && typeof value.name === 'string' ) {            // Reconstruct File as Blob (Files survive .entries() in Node.js)
            const blob = new Blob([await value.arrayBuffer()], { type: value.type });
            externalFormData.append(key, blob, value.name);
          } else {
            externalFormData.append(key, value);
          }
        }
        body = externalFormData;
        // 保留 X-TENANT-ID header
        const tenantId = request.headers.get('X-TENANT-ID');
        headers = new Headers();
        if (tenantId) {
          headers.set('X-TENANT-ID', tenantId);
        }
    } else {
      // JSON 或其他：读取为 text，再传给 fetch
      const text = await request.text();
      body = text;
      // 如果是 JSON，确保 content-type 正确
      if (contentType && contentType.includes('application/json') && text) {
        try {
          JSON.parse(text); // 验证 JSON
        } catch (e) {
          return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
        }
      }
    }
  }

  // 创建 AbortController 用于超时控制
  // 对于可能返回流式响应或需要LLM调用的请求，使用更长的超时时间
  const defaultTimeoutMs = parseInt(process.env.PROXY_TIMEOUT_MS || '60000', 10); // 默认 60 秒
  const streamingTimeoutMs = parseInt(process.env.PROXY_STREAMING_TIMEOUT_MS || '300000', 10); // 流式响应默认 5 分钟
  
  // 判断是否是可能返回流式响应或需要LLM调用的请求路径
  // 包括：
  // - /threads/* 下的所有路径（可能涉及LLM调用，如生成标题、消息等）
  // - /chat/completions 和 /chat（流式响应）
  const isPotentialStreamingPath = pathname.includes('/threads/') ||
                                    pathname.includes('/chat/completions') ||
                                    pathname.includes('/chat');
  
  // 对于可能返回流式响应或需要LLM调用的请求，使用更长的超时时间
  const initialTimeoutMs = isPotentialStreamingPath ? streamingTimeoutMs : defaultTimeoutMs;
  
  const controller = new AbortController();
  let timeoutId: NodeJS.Timeout | null = null;

  try {
    // 根据请求路径设置初始超时时间
    timeoutId = setTimeout(() => controller.abort(), initialTimeoutMs);

    const res = await fetch(upstreamUrl.toString(), {
      method,
      headers,
      body,
      signal: controller.signal,
      // 添加 keepalive 选项，保持连接活跃
      keepalive: true,
    });

    // 检查是否是流式响应
    const contentType = res.headers.get('content-type') || '';
    const isStreaming = contentType.includes('text/event-stream') || 
                        contentType.includes('stream') ||
                        res.headers.get('transfer-encoding') === 'chunked';

    if (isStreaming && res.body) {
      // 流式响应：清除当前超时，使用更长的超时时间
      if (timeoutId) clearTimeout(timeoutId);
      
      // 对于流式响应，创建一个新的超时控制器，使用更长的超时时间
      // 注意：这里我们不能直接修改signal，但可以在流式传输过程中监控
      // 实际上，对于流式响应，我们应该让客户端控制超时，而不是在代理层强制超时
      // 流式响应：直接传递流，不设置超时限制（由客户端或Next.js处理）
      return new NextResponse(res.body, {
        status: res.status,
        statusText: res.statusText,
        headers: res.headers,
      });
    }

    // 非流式响应：清除超时（响应已完全接收）
    if (timeoutId) clearTimeout(timeoutId);

    // 检查响应是否正常
    if (!res.ok && !res.body) {
      return NextResponse.json(
        { 
          error: 'Proxy request failed', 
          message: `Backend returned status ${res.status} without body`,
        }, 
        { status: res.status }
      );
    }

    // 非流式响应：读取完整数据
    const responseData = await res.blob(); // 通用处理（支持 JSON、text、binary）
    const responseHeaders = new Headers(res.headers);
    responseHeaders.set('content-length', responseData.size.toString());

    // 避免重复设置
    responseHeaders.delete('transfer-encoding');

    return new NextResponse(responseData, {
      status: res.status,
      statusText: res.statusText,
      headers: responseHeaders,
    });
  } catch (error: any) {
    if (timeoutId) clearTimeout(timeoutId);
    
    // 记录错误详情用于调试
    console.error("Proxy request failed: ", {
      name: error.name,
      message: error.message,
      code: error.code,
      cause: error.cause,
      stack: error.stack
    });
    
    // 处理连接关闭错误
    if (error.cause?.code === 'UND_ERR_SOCKET' || 
        error.message?.includes('other side closed') ||
        error.message?.includes('fetch failed') ||
        error.message?.includes('ECONNREFUSED') ||
        error.message?.includes('ENOTFOUND')) {
      return NextResponse.json(
        { 
          error: 'Proxy connection closed', 
          message: 'Backend connection was closed unexpectedly. This may happen if the request takes too long or the backend service restarted.',
          details: error.cause?.message || error.message,
          code: error.cause?.code || 'CONNECTION_CLOSED'
        }, 
        { status: 502 } // Bad Gateway - 后端服务问题
      );
    }
    
    // 处理超时错误（包括AbortError）
    if (error.name === 'AbortError' || 
        error.code === 'UND_ERR_HEADERS_TIMEOUT' ||
        error.code === 20 || // DOMException.ABORT_ERR
        error.message?.includes('aborted') ||
        error.message?.includes('This operation was aborted')) {
      return NextResponse.json(
        { 
          error: 'Proxy request timeout', 
          message: `Request exceeded timeout of ${initialTimeoutMs}ms. ${isPotentialStreamingPath ? 'This is a streaming endpoint, which may take longer to respond.' : 'Please try again or contact support if the issue persists.'}`,
          details: error.message,
          code: 'TIMEOUT'
        }, 
        { status: 504 }
      );
    }
    
    // 处理其他错误
    return NextResponse.json(
      { 
        error: 'Proxy request failed', 
        message: error.message || String(error),
        details: error.cause?.message || error.stack,
        code: error.code || 'UNKNOWN_ERROR'
      }, 
      { status: 500 }
    );
  }
}