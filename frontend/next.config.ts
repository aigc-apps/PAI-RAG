import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
    eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
  async rewrites() {
    if (process.env.NODE_ENV === 'development') {
      return [
        // 代理所有以/api开头的请求到目标服务器
        {
          source: '/v1/:path*', // 客户端请求的路径
          destination: 'http://localhost:8682/v1/:path*' // 代理目标地址
        }
      ];
    }
    else {
      console.log("not using rewrite in production.")
      return [];
    }
  }
};

export default nextConfig;
