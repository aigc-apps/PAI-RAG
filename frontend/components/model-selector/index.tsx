"use client";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Command,
  CommandGroup,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { useEffect, useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react";

interface ModelConfigurationParams {
  id: string;
  model_id: string;
}

interface ModelGroup {
  id: string;
  label: string;
  models: ModelConfigurationParams[];
}

interface ModelSelectorProps {
  selectedModel: {
    source: string;
    model_id: string;
  };
  onModelChange: (id: string, source: string, model_id: string) => void;
}

export default function ModelSelector({
  selectedModel,
  onModelChange,
}: ModelSelectorProps) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [modelGroups, setModelGroups] = useState<ModelGroup[]>([]);

  // 获取模型数据
  useEffect(() => {
    const fetchModels = async () => {
      setLoading(true);
      setError(null);
      try {
        const API_BASE =
          process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8680";
        const res = await fetch(`${API_BASE}/v1/config/llms/groups`);
        if (!res.ok) throw new Error("模型数据加载失败");
        const data = await res.json();
        console.log("model data: ", data);
        setModelGroups(data.groups);
      } catch (err) {
        setError("无法加载模型列表，请检查网络或服务状态");
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    if (open) {
      fetchModels();
    }
  }, [open]);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger className="min-w-[180px] w-[250px] bg-transparent shadow-none focus:outline-none cursor-pointer hover:bg-gray-100 rounded transition-colors border-none text-gray-600 h-9 px-3 py-2 text-sm focus:ring-1 focus:ring-ring">
        <div className="flex items-center pr-2 truncate">
          <span className="flex flex-row items-center justify-start gap-2 text-lg font-semibold">
            {selectedModel.model_id || "请选择模型"}
          </span>
          <ChevronsUpDown className="size-4 opacity-50 ml-auto" />
        </div>
      </PopoverTrigger>
      <PopoverContent className="min-w-[180px] w-[280px] p-0 shadow-md rounded-md">
        <Command>
          <CommandList>
            {loading ? (
              <div className="p-4 text-center text-sm text-gray-500">
                加载中...
              </div>
            ) : error ? (
              <div className="p-4 text-center text-sm text-red-500">
                {error}
              </div>
            ) : (
              modelGroups.map((group) => (
                <CommandGroup
                  key={group.id}
                  heading={group.label}
                  className="w-full"
                >
                  {group.models.map((model) => (
                    <CommandItem
                      key={model.model_id}
                      value={model.model_id}
                      onSelect={() => {
                        onModelChange(model.id, group.id, model.model_id);
                        setOpen(false);
                      }}
                      className="flex items-center"
                    >
                      <Check
                        className={`mr-1 size-4 ${
                          selectedModel.model_id === model.model_id
                            ? "opacity-100"
                            : "opacity-0"
                        }`}
                      />
                      <span className="flex flex-row w-full items-center justify-start gap-2">
                        {model.model_id}
                      </span>
                    </CommandItem>
                  ))}
                </CommandGroup>
              ))
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
