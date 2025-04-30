import React, { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import {
  SaveIcon,
  PlusIcon,
  TrashIcon,
  ChevronDownIcon,
  SettingsIcon,
} from "lucide-react";
import { motion } from "framer-motion";

export default function MCPConfigPage() {
  const [mcps, setMCPs] = useState([
    { id: 1, name: "", url: "", type: "", active: false },
  ]);

  // 页面加载时从后端拉取配置
  useEffect(() => {
    fetchConfigs();
  }, []);

  // 拉取配置
  const fetchConfigs = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/configs");
      if (res.ok) {
        const data = await res.json();
        setMCPs(data || []);
      } else {
        setMCPs([]); // 文件不存在时初始化为空数组
      }
    } catch (error) {
      console.error("拉取配置失败:", error);
    }
  };

  // 保存配置到后端
  const saveToServer = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/configs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ configs: mcps }),
      });

      if (!res.ok) throw new Error("保存失败");
      alert("配置已保存至本地文件");
    } catch (err) {
      console.error(err);
      alert("保存失败，请重试");
    }
  };

  // 添加新配置
  const addMCP = () => {
    setMCPs([
      ...mcps,
      { id: Date.now(), name: "", url: "", type: "", active: false },
    ]);
  };

  // 删除指定配置
  const removeMCP = (id: number) => {
    setMCPs(mcps.filter((mcp) => mcp.id !== id));
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6 px-6">
        <h1 className="text-3xl font-bold text-gray-800 shrink-0">
          MCP 配置中心
        </h1>
        <div className="flex flex-col gap-2 ml-auto">
          <Button
            onClick={addMCP}
            variant="secondary"
            className="flex items-center gap-2"
          >
            <PlusIcon className="w-4 h-4" />
            添加新服务器
          </Button>
          <Button onClick={saveToServer} className="flex items-center gap-2">
            <SaveIcon className="w-4 h-4 mr-2" />
            保存配置
          </Button>
        </div>
      </div>

      {/* 卡片容器 */}
      <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-2 gap-6">
        {mcps.map((mcp) => (
          <motion.div
            key={mcp.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            <Card className="overflow-hidden border border-gray-200 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300">
              <div className="p-5 bg-gradient-to-r from-gray-50 to-gray-100 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-gray-700">
                    {mcp.name || "未命名服务器"}
                  </span>
                  <div className="flex items-center gap-2">
                    <span
                      className={`px-2 py-1 text-xs rounded-full ${
                        mcp.active
                          ? "bg-green-100 text-green-800"
                          : "bg-gray-100 text-gray-800"
                      }`}
                    >
                      {mcp.active ? "已启用" : "已停用"}
                    </span>
                    <button
                      onClick={() => removeMCP(mcp.id)}
                      className="text-red-500 hover:text-red-700 transition-colors p-1 rounded-full hover:bg-red-50"
                    >
                      <TrashIcon className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              </div>

              <div className="p-5">
                <div className="space-y-4">
                  {/* 名称输入 */}
                  <div>
                    <label className="block text-sm font-medium text-gray-500 mb-1">
                      服务器名称
                    </label>
                    <Input
                      placeholder="请输入服务器名称"
                      value={mcp.name}
                      onChange={(e) => {
                        const updated = mcps.map((item) =>
                          item.id === mcp.id
                            ? { ...item, name: e.target.value }
                            : item,
                        );
                        setMCPs(updated);
                      }}
                      className="w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>

                  {/* URL 输入 */}
                  <div>
                    <label className="block text-sm font-medium text-gray-500 mb-1">
                      服务器地址
                    </label>
                    <Input
                      placeholder="https://your-mcp-server.com/sse"
                      value={mcp.url}
                      onChange={(e) => {
                        const updated = mcps.map((item) =>
                          item.id === mcp.id
                            ? { ...item, url: e.target.value }
                            : item,
                        );
                        setMCPs(updated);
                      }}
                      className="w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>

                  {/* Transport 类型 */}
                  <div>
                    <label className="block text-sm font-medium text-gray-500 mb-1">
                      传输协议
                    </label>
                    <div className="relative">
                      <select
                        value={mcp.type}
                        onChange={(e) => {
                          const updated = mcps.map((item) =>
                            item.id === mcp.id
                              ? { ...item, type: e.target.value }
                              : item,
                          );
                          setMCPs(updated);
                        }}
                        className="w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500 appearance-none bg-white pr-8"
                      >
                        <option value="sse">SSE</option>
                        <option value="stdio">Stdio</option>
                      </select>
                      <ChevronDownIcon className="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                    </div>
                  </div>

                  {/* 激活状态 */}
                  <div className="pt-2">
                    <div className="flex items-center space-x-3">
                      <input
                        id={`active-${mcp.id}`}
                        type="checkbox"
                        checked={mcp.active}
                        onChange={(e) => {
                          const updated = mcps.map((item) =>
                            item.id === mcp.id
                              ? { ...item, active: e.target.checked }
                              : item,
                          );
                          setMCPs(updated);
                        }}
                        className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <label
                        htmlFor={`active-${mcp.id}`}
                        className="text-sm font-medium"
                      >
                        是否激活
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* 空状态提示 */}
      {mcps.length === 0 && (
        <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-gray-200 rounded-xl bg-gray-50">
          <SettingsIcon className="w-12 h-12 text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-700">暂无 MCP 服务器</h3>
          <p className="text-gray-500 mt-1">
            点击下方按钮添加新的 MCP 服务器配置
          </p>
          <Button
            onClick={addMCP}
            className="mt-4 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
          >
            添加服务器
          </Button>
        </div>
      )}

      {/* <h1 className="text-2xl mb-4">MCP 配置</h1>
      <div className="flex flex-wrap gap-4">
        {mcps.map((mcp) => (
          <Card key={mcp.id} className="w-[300px] p-4">
            <div className="space-y-2">
            MCP Server 名称  <Input
                placeholder="MCP Server 名称"
                value={mcp.name}
                onChange={(e) => {
                  const updated = mcps.map(item =>
                    item.id === mcp.id ? { ...item, name: e.target.value } : item
                  );
                  setMCPs(updated);
                }}
              />
              URL <Input
                placeholder="MCP Server URL"
                value={mcp.url}
                onChange={(e) => {
                  const updated = mcps.map(item =>
                    item.id === mcp.id ? { ...item, url: e.target.value } : item
                  );
                  setMCPs(updated);
                }}
              />
              <label>Transport</label>
              <Input
                placeholder="MCP Server Transport"
                value={mcp.type}
                onChange={(e) => {
                  const updated = mcps.map(item =>
                    item.id === mcp.id ? { ...item, type: e.target.value } : item
                  );
                  setMCPs(updated);
                }}
              />
              <div className="flex items-center space-x-2">

                <input
                  id={`active-${mcp.id}`}
                  type="checkbox"
                  checked={mcp.active}
                  onChange={(e) => {
                    const updated = mcps.map(item =>
                      item.id === mcp.id ? { ...item, active: e.target.checked } : item
                    );
                    setMCPs(updated);
                  }}
                  className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <label htmlFor={`active-${mcp.id}`} className="text-sm font-medium">
                  是否激活
                </label>
              </div>
              <Button onClick={() => removeMCP(mcp.id)} variant="outline" className="w-full">
                删除
              </Button>
            </div>
          </Card>
        ))}
      </div> */}
      {/* <Button onClick={addMCP} className="mt-4">+ 添加新 MCP Server</Button>
      <Button onClick={saveToServer}>保存到后端</Button> */}
    </div>
  );
}
