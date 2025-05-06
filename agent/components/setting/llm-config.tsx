import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { useState } from "react"
import { Card } from "@/components/ui/card";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
  } from "@/components/ui/select"
import {
    PlusIcon,
    TrashIcon,
    SettingsIcon,
  } from "lucide-react";
import { motion } from "framer-motion";


interface LLMConfigProps {
    config: Array<{
      id: number;
      source: string;
      model_name: string;
      api_key: string;
      max_context: number;
    }>;
    onChange: (updatedConfig: Array<{
        id: number;
        source: string;
        model_name: string;
        api_key: string;
        max_context: number;
      }>) => void;
  }
const LLMConfig: React.FC<LLMConfigProps> = ({ config, onChange })  => {
    const [isOpen, setIsOpen] = useState(false)

    // 添加新配置
    const addLLM = () => {
        const newLLM = { id: Date.now(), source: "", model_name: "", api_key: "", max_context: 0};
        onChange([...config, newLLM]);
    };

    // 删除指定配置
    const removeLLM = (id: number) => {
        onChange(config.filter((llm) => llm.id !== id));
    };
  

    return (
        <div className={`transition-colors rounded-lg p-4 overflow-hidden duration-200 ${isOpen ? 'bg-gray-100' : ''}`}>
            <Collapsible open={isOpen} onOpenChange={setIsOpen} >
                <CollapsibleTrigger asChild>
                    <Button variant="ghost" className="w-full justify-start items-center">
                        {isOpen ? "▼ LLM 配置" : "▶ LLM 配置"}
                    </Button>
                </CollapsibleTrigger>
                <CollapsibleContent
                    className="transition-all data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down"
                >
                    <div className="space-y-4 mt-4">
                        <div className="flex flex-row gap-2 ml-auto">
                            <Button
                                onClick={addLLM}
                                className="flex items-center gap-2"
                            >
                                <PlusIcon className="w-4 h-4" />
                                添加
                            </Button>
                        </div>

                        {/* 卡片容器 */}
                <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-2 gap-6">
                  {config.map((llm) => (
                    <motion.div
                        key={llm.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        transition={{ duration: 0.2 }}
                    >
                        <Card className="overflow-hidden border border-gray-200 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300">
                        <div className="p-5 bg-gradient-to-r from-gray-50 to-gray-100 border-b border-gray-200">
                            <div className="flex items-center justify-between">
                            <span className="font-semibold text-gray-700">
                                {llm.model_name || "未命名模型"}
                            </span>
                            <div className="flex items-center gap-2">
                                
                                <button
                                onClick={() => removeLLM(llm.id)}
                                className="text-red-500 hover:text-red-700 transition-colors p-1 rounded-full hover:bg-red-50"
                                >
                                <TrashIcon className="w-5 h-5" />
                                </button>
                            </div>
                            </div>
                        </div>

                        <div className="p-5">
                            <div className="space-y-4">
                            {/* 模型的来源输入 */}
                            <div>
                                <label className="block text-sm font-medium text-gray-500 mb-1">
                                模型来源名称
                                </label>
                                <Select
                                    value={llm.source} // 绑定当前来源
                                    onValueChange={(value) => {
                                    const updated = config.map((item) =>
                                        item.id === llm.id ? { ...item, source: value } : item
                                    );
                                    onChange(updated); // 触发父组件更新
                                    }}
                                >
                                    <SelectTrigger className="w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500">
                                    <SelectValue placeholder="请选择模型来源" />
                                    </SelectTrigger>
                                    <SelectContent>
                                    <SelectItem value="openai">OpenAI</SelectItem>
                                    <SelectItem value="qwen">Qwen</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>

                            {/* 模型名称输入 */}
                            <div>
                                <label className="block text-sm font-medium text-gray-500 mb-1">
                                模型名称
                                </label>
                                <Input
                                placeholder="请输入模型名称"
                                value={llm.model_name}
                                onChange={(e) => {
                                  const updated = config.map((item) =>
                                    item.id === llm.id ? { ...item, model_name: e.target.value } : item
                                  );
                                  onChange(updated); // 触发父组件更新
                                }}
                                className="w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                                />
                            </div>
                            
                            {/* API Key输入 */}
                            <div>
                                <label className="block text-sm font-medium text-gray-500 mb-1">
                                API Key
                                </label>
                                <Input
                                placeholder="请输入API Key"
                                value={llm.api_key}
                                onChange={(e) => {
                                  const updated = config.map((item) =>
                                    item.id === llm.id ? { ...item, api_key: e.target.value } : item
                                  );
                                  onChange(updated); // 触发父组件更新
                                }}
                                className="w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                                />
                            </div>

                            {/* 模型上下文长度 */}
                            {/* <div>
                                <label className="block text-sm font-medium text-gray-500 mb-1">
                                模型上下文长度
                                </label>
                                <Input
                                    value={llm.max_context}
                                    type="number"
                                    onChange={(e) => {
                                      const updated = config.map((item) =>
                                        item.id === llm.id ? { ...item, max_context: Number(e.target.value) } : item
                                      );
                                      onChange(updated); // 触发父组件更新
                                    }}
                                    className="w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                                />
                            </div> */}

                            </div>
                        </div>
                        </Card>
                    </motion.div>
                    ))}
                </div>

                {/* 空状态提示 */}
                {config.length === 0 && (
                    <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-gray-200 rounded-xl bg-gray-50">
                    <SettingsIcon className="w-12 h-12 text-gray-400 mb-4" />
                    <h3 className="text-lg font-medium text-gray-700">暂无 LLM</h3>
                    <p className="text-gray-500 mt-1">
                        点击下方按钮添加新的 LLM
                    </p>
                    <Button
                        onClick={addLLM}
                        // className="mt-4 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
                    >
                        添加LLM
                    </Button>
                    </div>
                )}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
    )
}

export default LLMConfig