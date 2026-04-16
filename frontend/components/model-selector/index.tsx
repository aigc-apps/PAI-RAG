'use client';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { useEffect, useState } from 'react';
import { Check, ChevronsUpDown, Plus, Bot } from 'lucide-react';
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

        if (!appRes.ok) throw new Error(t('common.loadModelFailed'));
        const chatbotData = (await appRes.json()).data.items;
        const chatbotGroup: ModelGroup = {
          id: 'chatbot',
          label: t('common.chatApplication'),
          models: chatbotData.map((item: any) => ({
            id: item.id,
            model_id: item.app_id,
          })),
        };
        const groups = [chatbotGroup, ...data.data.groups];
        setModelGroups(groups);

        let foundModel: ModelConfigurationParams | null = null;
        let foundGroup: ModelGroup | null = null;
        if (selectedModel.model_id) {
          for (const group of groups) {
            const model = group.models.find(
              (m: ModelConfigurationParams) => m.model_id === selectedModel.model_id,
            );
            if (model) {
              foundModel = model;
              foundGroup = group;
              break;
            }
          }
        }

        if (foundModel && foundGroup) {
          setCurrentModel(foundModel.model_id);
          onModelChange(foundModel.id, foundGroup.id, foundModel.model_id);
        } else {
          for (const group of groups) {
            if (group.models.length > 0) {
              const firstModel = group.models[0];
              setCurrentModel(firstModel.model_id);

              onModelChange(firstModel.id, group.id, firstModel.model_id);
              break;
            }
          }
          if (groups.every((g) => g.models.length === 0)) {
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
      <PopoverTrigger
        className="inline-flex items-center gap-1.5 px-2.5 py-1 h-7 max-w-[260px] rounded-md text-xs font-medium text-foreground bg-primary/[0.06] border border-primary/20 hover:bg-primary/10 hover:border-primary/30 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
      >
        <Bot className="w-3.5 h-3.5 text-primary shrink-0" />
        <span className="truncate" suppressHydrationWarning>
          {currentModel || t('common.selectModel')}
        </span>
        <ChevronsUpDown className="w-3 h-3 text-primary/60 shrink-0 ml-0.5" />
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-64 p-0 shadow-lg rounded-lg overflow-hidden"
      >
        <div className="max-h-[420px] overflow-y-auto py-1">
          {loading ? (
            <div className="py-6 text-center text-xs text-muted-foreground">
              {t('common.loading')}
            </div>
          ) : error ? (
            <div className="py-4 px-3 text-center text-xs">
              <p className="text-destructive mb-2">{error}</p>
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
            modelGroups.map((group, idx) => {
              if (group.models.length === 0) return null;
              return (
                <div key={group.id}>
                  {idx > 0 && <div className="my-1 border-t border-border/60" />}
                  <div className="px-2.5 pt-1.5 pb-1 flex items-center gap-1.5">
                    <Bot className="w-3 h-3 text-muted-foreground" />
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      {group.label}
                    </span>
                  </div>
                  <div className="px-1">
                    {group.models.map((model) => {
                      const isSelected = selectedModel.model_id === model.model_id;
                      return (
                        <button
                          key={model.model_id}
                          type="button"
                          onClick={() => {
                            onModelChange(model.id, group.id, model.model_id);
                            setOpen(false);
                            setCurrentModel(model.model_id);
                                        }}
                          className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-colors ${
                            isSelected
                              ? 'bg-primary/10 text-primary font-medium'
                              : 'hover:bg-muted text-foreground/90'
                          }`}
                        >
                          <Check
                            className={`w-3.5 h-3.5 shrink-0 ${
                              isSelected ? 'opacity-100 text-primary' : 'opacity-0'
                            }`}
                          />
                          <span className="truncate flex-1 text-left">
                            {model.model_id}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
