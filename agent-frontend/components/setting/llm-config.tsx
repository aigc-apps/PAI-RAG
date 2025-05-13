import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
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
import { Separator } from "@/components/ui/separator";


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
const LLMConfig: React.FC<LLMConfigProps> = ({ config, onChange }) => {

    // 添加新配置
    const addLLM = () => {
        const newLLM = { id: Date.now(), source: "", model_name: "", api_key: "", max_context: 0 };
        onChange([...config, newLLM]);
    };

    // 删除指定配置
    const removeLLM = (id: number) => {
        onChange(config.filter((llm) => llm.id !== id));
    };


    return (
        <div className={`transition-colors rounded-lg overflow-hidden`}>
            <Button
                onClick={addLLM}
                className="flex items-center gap-2"
            >
                <PlusIcon className="w-4 h-4" />
                添加
            </Button>
            <Separator className="my-2" />
            <div className="grid gap-4">
                {config.map((llm) => (
                    <motion.div
                        key={llm.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        transition={{ duration: 0.2 }}
                    >
                        <div className="grid grid-cols-7 gap-2 bg-white hover:shadow-sm">
                            <div className="col-span-2">
                                <Input
                                    placeholder="模型名称"
                                    value={llm.model_name}
                                    onChange={(e) => {
                                        const updated = config.map((item) =>
                                            item.id === llm.id ? { ...item, model_name: e.target.value } : item
                                        );
                                        onChange(updated);
                                    }}
                                    className="text-sm"
                                />
                            </div>

                            <div className="col-span-2">
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

                            <div className="col-span-2">
                                <Input
                                    placeholder="API Key"
                                    value={llm.api_key}
                                    onChange={(e) => {
                                        const updated = config.map((item) =>
                                            item.id === llm.id ? { ...item, api_key: e.target.value } : item
                                        );
                                        onChange(updated);
                                    }}
                                    className="text-sm"
                                />

                            </div>
                            <div className="col-span-1 relative">
                                <button
                                    onClick={() => removeLLM(llm.id)}
                                    className="absolute right-2 top-1/2 transform -translate-y-1/2 text-red-500 hover:text-red-700"
                                >
                                    <TrashIcon className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                        <Separator className="my-2" />

                    </motion.div>
                ))}
            </div>
            {config.length === 0 && (
                <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-gray-200 rounded-xl bg-gray-50">
                    <SettingsIcon className="w-10 h-6 text-gray-400 mb-4" />
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
    )
}

export default LLMConfig