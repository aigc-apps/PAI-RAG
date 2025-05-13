import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Button } from "@/components/ui/button"
import React, { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  SaveIcon,
  PlusIcon,
  TrashIcon,
  SettingsIcon,
} from "lucide-react";
import { motion } from "framer-motion";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Separator } from "@/components/ui/separator"
import { stringify } from "querystring";
import { StringDecoder } from "string_decoder";

interface MCPConfigProps {
  config: Array<{
    id: number;
    name: string;
    url: string;
    type: string;
    active: boolean;
  }>;
  onChange: (updatedConfig: Array<{
    id: number;
    name: string;
    url: string;
    type: string;
    active: boolean;
  }>) => void;
}

const MCPConfig: React.FC<MCPConfigProps> = ({ config, onChange }) => {
  // 添加新配置
  const addMCP = () => {
    const newMCP = { id: Date.now(), name: "", url: "", type: "", active: false };
    onChange([...config, newMCP]);
  };

  // 删除指定配置
  const removeMCP = (id: number) => {
    onChange(config.filter((mcp) => mcp.id !== id));
  };

  return (
    <div className={`transition-colors rounded-lg p-4 overflow-hidden duration-200`}>
      <Button
        onClick={addMCP}
        className="flex items-center gap-2"
      >
        <PlusIcon className="w-4 h-4" />
        添加
      </Button>

      <Separator className="my-2" />
      <div className="grid gap-6">
      <Accordion type="single" collapsible>
        {config.map((mcp) => (
          <motion.div
            key={mcp.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            <AccordionItem value={mcp.id.toString()}>
              <AccordionTrigger>
                <div className="grid grid-cols-8">
                  <div className="col-span-4">
                    <span className="font-semibold text-gray-700">
                      {mcp.name || "未命名服务器"}
                    </span>
                  </div>
                  <div className="col-span-2 ">
                    <span
                      className={`px-2 py-2 text-xs rounded-full ${mcp.active
                          ? "bg-green-100 text-green-800"
                          : "bg-gray-100 text-gray-800"
                        }`}
                    >
                      {mcp.active ? "已启用" : "已停用"}
                    </span>
                  </div>
                  <div className="col-span-2">
                    <TrashIcon className="w-5 h-5 text-red-500 hover:text-red-700 hover:bg-red-50" onClick={() => removeMCP(mcp.id)}/>
                  </div>
                </div>
              </AccordionTrigger>
              <AccordionContent>
                <div className="grid grid-cols-8 gap-2 bg-white hover:shadow-sm">
                    <div className="col-span-2">
                      <Input
                        placeholder="请输入服务器名称"
                        value={mcp.name}
                        onChange={(e) => {
                          const updated = config.map((item) =>
                            item.id === mcp.id ? { ...item, name: e.target.value } : item
                          );
                          onChange(updated); // 触发父组件更新
                        }}
                        className="w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                      />
                    </div>
                    <div className="col-span-2">
                      <Input
                        placeholder="https://your-mcp-server.com/sse"
                        value={mcp.url}
                        onChange={(e) => {
                          const updated = config.map((item) =>
                            item.id === mcp.id ? { ...item, url: e.target.value } : item
                          );
                          onChange(updated); // 触发父组件更新
                        }}
                        className="w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                      />
                    </div>
                    <div className="col-span-2">
                      <Input
                        placeholder="sse"
                        value={mcp.type}
                        onChange={(e) => {
                          const updated = config.map((item) =>
                            item.id === mcp.id ? { ...item, type: e.target.value } : item
                          );
                          onChange(updated); // 触发父组件更新
                        }}
                        className="w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                      />
                    </div>
                    <div className="col-span-2">
                      <input
                          id={`active-${mcp.id}`}
                          type="checkbox"
                          checked={mcp.active}
                          onChange={(e) => {
                            const updated = config.map((item) =>
                              item.id === mcp.id ? { ...item, active: e.target.checked } : item
                            );
                            onChange(updated); // 触发父组件更新
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
              </AccordionContent>
            </AccordionItem>
            <Separator className="my-2" />
          </motion.div>
        ))}
      </Accordion>
      </div>

      {/* 空状态提示 */}
      {config.length === 0 && (
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
    </div>
  )
}

export default MCPConfig