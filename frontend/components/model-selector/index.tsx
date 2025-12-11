'use client';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import {
  Command,
  CommandGroup,
  CommandItem,
  CommandList,
} from '@/components/ui/command';
import { useEffect, useState } from 'react';
import { Check, ChevronsUpDown, Plus } from 'lucide-react';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import Link from 'next/link';

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
    model_id: string | undefined;
  };
  onModelChange: (id: string, source: string, model_id: string) => void;
}

export default function ModelSelector({
  selectedModel,
  onModelChange,
}: ModelSelectorProps) {
  const [currentModel, setCurrentModel] = useState(selectedModel.model_id);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [modelGroups, setModelGroups] = useState<ModelGroup[]>([]);
  const { tenantFetch } = useTenantFetch();

  // 获取模型数据
  useEffect(() => {
    const fetchModels = async () => {
      setLoading(true);
      setError(null);
      try {
        const [llmRes, appRes] = await Promise.all([
          tenantFetch(`/api/config/llms/groups`),
          tenantFetch(`/api/config/apps`),
        ]);
        if (!llmRes.ok) throw new Error('模型数据加载失败');
        const data = await llmRes.json();
        console.log('model data: ', data);

        if (!appRes.ok) throw new Error('模型数据加载失败');
        const chatbotData = (await appRes.json()).data.items;
        const chatbotGroup = {
          id: 'chatbot',
          label: '对话应用',
          models: chatbotData.map((item: any) => {
            return {
              id: item.id,
              model_id: item.app_id,
              group_id: 'chatbot',
            };
          }),
        };
        const modelGroups = [chatbotGroup, ...data.data.groups];
        setModelGroups(modelGroups);

        // 查找当前选中的模型是否存在
        let foundModel: ModelConfigurationParams | null = null;
        let foundGroup: ModelGroup | null = null;
        if (selectedModel.model_id) {
          for (const group of modelGroups) {
            const model = group.models.find((m: ModelConfigurationParams) => m.model_id === selectedModel.model_id);
            if (model) {
              foundModel = model;
              foundGroup = group;
              break;
            }
          }
        }

        if (foundModel && foundGroup) {
          // 找到了保存的模型，使用它
          setCurrentModel(foundModel.model_id);
          onModelChange(foundModel.id, foundGroup.id, foundModel.model_id);
        } else {
          // 没有选中的模型或保存的模型不存在，选择第一个可用的模型
          let found = false;
          for (const group of modelGroups) {
            if (group.models.length > 0) {
              const firstModel = group.models[0];
              setCurrentModel(firstModel.model_id);
              onModelChange(firstModel.id, group.id, firstModel.model_id);
              found = true;
              break;
            }
          }
          if (!found) {
            setError('没有可用的模型');
          }
        }
      } catch (err) {
        setError('无法加载模型列表，请检查网络或服务状态');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, [open]);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger className="absolute top-2 left-12 max-w-[250px] bg-transparent shadow-none focus:outline-none cursor-pointer hover:bg-gray-100 rounded transition-colors border-none text-gray-600 text-sm focus:ring-1 focus:ring-ring">
        <div className="flex items-center truncate gap-2">
          <span className="text-md font-medium items-center">
            {currentModel || '请选择模型'}
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
              <div className="p-4 text-center text-sm">
                <p className="text-red-500 mb-2">{error}</p>
                <Link 
                  href="/config/model" 
                  className="inline-flex items-center gap-1 text-primary hover:underline"
                  onClick={() => setOpen(false)}
                >
                  <Plus className="h-3 w-3" />
                  去添加模型
                </Link>
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
                        setCurrentModel(model.model_id);
                      }}
                      className="flex items-center"
                    >
                      <Check
                        className={`mr-1 size-4 ${
                          selectedModel.model_id === model.model_id
                            ? 'opacity-100'
                            : 'opacity-0'
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
