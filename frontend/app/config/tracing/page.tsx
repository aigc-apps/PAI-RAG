"use client";

import { Button } from "@/components/ui/button";
import React, { useState, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import * as Toast from "@radix-ui/react-toast";

export default function TracingConfig() {
  const [endpoint, setEndpoint] = useState("");
  const [token, setToken] = useState("");
  const [serviceName, setServiceName] = useState("");
  const [isLoading, setIsLoading] = useState(false); // 加载状态
  const [error, setError] = useState(""); // 错误提示
  const [toastState, setToastState] = useState({
    open: false,
    title: "",
    description: "",
    variant: "default" as "default" | "destructive",
  });

  // 初始化加载配置
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setIsLoading(true);
        setError("");

        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
        const res = await fetch(`http://localhost:${port}/v1/config/trace`, {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        });

        if (!res.ok) throw new Error("加载配置失败");

        const data = await res.json();
        setEndpoint(data["endpoint"] || "");
        setToken(data["token"] || "");
        setServiceName(data["service_name"] || "");
      } catch (err: any) {
        setError(err.message || "加载失败");
        setToastState({
          open: true,
          title: "配置加载失败",
          description: err.message || "请检查网络或重试",
          variant: "destructive",
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchConfig();
  }, []);
  // 保存配置
  const handleSave = async () => {
    if (!endpoint || !token || !serviceName) {
      setError("endpoint、token、serviceName 均不能为空");
      return;
    }
    try {
      setIsLoading(true);
      setError("");

      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
      const res = await fetch(`http://localhost:${port}/v1/config/trace`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          endpoint: endpoint,
          token: token,
          service_name: serviceName,
        }),
      });

      if (!res.ok) throw new Error("保存失败，请检查网络或配置");

      setToastState({
        open: true,
        title: "阿里云链路追踪配置已成功保存",
        description: "阿里云链路追踪配置已成功保存",
        variant: "default",
      });
    } catch (err: any) {
      setError(err.message || "保存失败，请重试");
      setToastState({
        open: true,
        title: "阿里云链路追踪配置保存失败",
        description: err.message || "请检查网络或重试",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div id="tracing">
      <div
        className={`transition-colors rounded-lg p-4 overflow-hidden duration-200`}
      >
        <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-gray-200 rounded-xl bg-gray-50">
          <h2 className="text-2xl font-bold text-gray-800">
            阿里云链路追踪配置
          </h2>
          <a
            href="https://help.aliyun.com/zh/opentelemetry/quick-start?spm=a2c4g.11186623.help-menu-90275.d_1.15c45dc7tG5ukV#prereq-3jq-3as-xo9"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline text-sm"
          >
            如何获取endpoint/token信息
          </a>

          <div className="grid gap-4 py-4 max-w-xl w-full mx-auto">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="endpoint">Endpoint</Label>
              <div className="col-span-3 flex items-center">
                <Input
                  id="endpoint"
                  value={endpoint}
                  onChange={(e) => setEndpoint(e.target.value)}
                  placeholder="输入 endpoint"
                  className="col-span-3"
                />
              </div>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="token">Token</Label>
              <div className="col-span-3 flex items-center">
                <Input
                  id="token"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder="输入 token"
                  className="col-span-3"
                />
              </div>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="serivceName">SerivceName</Label>
              <div className="col-span-3 flex items-center">
                <Input
                  id="serivceName"
                  value={serviceName}
                  onChange={(e) => setServiceName(e.target.value)}
                  placeholder="输入 serivceName"
                  className="col-span-3"
                />
              </div>
            </div>
          </div>
          <Button
            onClick={handleSave}
            disabled={isLoading}
            className="mt-4 px-4 py-2 text-white rounded-lg transition-colors"
          >
            {isLoading ? "保存中..." : "保存链路追踪配置"}
          </Button>
          {error && <p className="text-red-500 mt-2">{error}</p>}
        </div>
        <Toast.Root
          open={toastState.open}
          onOpenChange={(open) => setToastState((prev) => ({ ...prev, open }))}
          className={`grid grid-cols-[auto_1fr] items-center gap-x-4 rounded-md border px-4 py-6 shadow-lg transition-all data-[state=open]:animate-slideIn data-[state=closed]:animate-fadeOut ${
            toastState.variant === "destructive"
              ? "border-red-500 bg-red-50 text-red-900"
              : "border-gray-200 bg-white text-gray-900"
          }`}
        >
          <Toast.Description className="pl-4 text-sm font-medium">
            {toastState.description}
          </Toast.Description>
          <Toast.Action
            altText="关闭"
            onClick={() => setToastState((prev) => ({ ...prev, open: false }))}
          >
            ×
          </Toast.Action>
        </Toast.Root>

        {/* 触发 Toast 的隐藏容器 */}
        <Toast.Viewport className="fixed bottom-0 right-0 z-[100] m-0 flex w-96 flex-col gap-2 p-6" />
      </div>
    </div>
  );
}
