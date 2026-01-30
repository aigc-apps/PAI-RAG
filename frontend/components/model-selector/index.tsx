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
import { useI18n } from '@/app/providers/i18n';

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
  const { t } = useI18n();
  const [currentModel, setCurrentModel] = useState(selectedModel.model_id);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [modelGroups, setModelGroups] = useState<ModelGroup[]>([]);
  const { tenantFetch } = useTenantFetch();

  // Fetch model data
  useEffect(() => {
    const fetchModels = async () => {
      setLoading(true);
      setError(null);
      try {
        const [llmRes, appRes] = await Promise.all([
          tenantFetch(`/api/config/llms/groups`),
          tenantFetch(`/api/config/apps`),
        ]);
        if (!llmRes.ok) throw new Error(t('common.loadModelFailed'));
        const data = await llmRes.json();
        console.log('model data: ', data);

        if (!appRes.ok) throw new Error(t('common.loadModelFailed'));
        const chatbotData = (await appRes.json()).data.items;
        const chatbotGroup = {
          id: 'chatbot',
          label: t('common.chatApplication'),
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

        // Find if the currently selected model exists
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
          // Found the saved model, use it
          setCurrentModel(foundModel.model_id);
          onModelChange(foundModel.id, foundGroup.id, foundModel.model_id);
        } else {
          // No selected model or saved model doesn't exist, select the first available model
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
            setError(t('common.noAvailableModel'));
          }
        }
      } catch (err) {
        setError(t('common.loadModelFailed'));
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, [open]);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger className="absolute top-4 left-12 max-w-[250px] bg-transparent shadow-none focus:outline-none cursor-pointer hover:bg-gray-100 rounded transition-colors border-none text-gray-600 text-sm focus:ring-1 focus:ring-ring">
        <div className="flex items-center truncate gap-2">
          <span className="text-md font-medium items-center">
            {currentModel || t('common.selectModel')}
          </span>
          <ChevronsUpDown className="size-4 opacity-50 ml-auto" />
        </div>
      </PopoverTrigger>
      <PopoverContent className="min-w-[180px] w-[280px] p-0 shadow-md rounded-md">
        <Command>
          <CommandList>
            {loading ? (
              <div className="p-4 text-center text-sm text-gray-500">
                {t('common.loading')}
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
                  {t('common.goAddModel')}
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
