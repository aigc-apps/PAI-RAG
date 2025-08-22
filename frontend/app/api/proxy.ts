// app/api/proxy/route.js
import { Header } from '@radix-ui/react-accordion';
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
          if (value instanceof File) {
            // Reconstruct File as Blob (Files survive .entries() in Node.js)
            const blob = new Blob([await value.arrayBuffer()], { type: value.type });
            externalFormData.append(key, blob, value.name);
          } else {
            externalFormData.append(key, value);
          }
        }
        body = externalFormData;
        headers = new Headers();
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

  try {
    const res = await fetch(upstreamUrl.toString(), {
      method,
      headers,
      body,
    });

    // 读取响应数据
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
  } catch (error) {
    console.log("Proxy request failed: ", error)
    return NextResponse.json({ error: 'Proxy request failed', message: error }, { status: 500 });
  }
}