// components/EvalConfigFormDialog.tsx
'use client';

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ChevronDownIcon } from "lucide-react";
import { Spinner } from "@/components/ui/loading";
import { useState, useEffect } from 'react';
import { LlmConfig } from '@/app/config/model/llm/page';
import { useRouter } from 'next/navigation';
import { EvaluatorConfig } from '@/app/evaluation/[datasetId]/types';
import { useI18n } from '@/app/providers/i18n';


interface EvalConfigFormDialogProps {
  mode: 'new' | 'edit';
  config?: EvaluatorConfig; // When editing, pass in config
  llms: LlmConfig[];
  datasetId: string;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (config: EvaluatorConfig) => void;
  isSaving: boolean;
}

export function EvalConfigFormDialog({
  mode,
  config,
  llms,
  datasetId,
  isOpen,
  onOpenChange,
  onSave,
  isSaving,
}: EvalConfigFormDialogProps) {
  const router = useRouter();
  const { t } = useI18n();
  const [localConfig, setLocalConfig] = useState<EvaluatorConfig>(
    mode === 'edit' && config
      ? { ...config }
      : {
        id: "",
        name: "",
        type: "",
        model_id: "",
        case_sensitive: false,
        ignore_punctuation: false
      }
  );

  // Reset form when config or mode changes
  useEffect(() => {
    if (mode === 'edit' && config) {
      setLocalConfig({ ...config });
    } else {
      setLocalConfig({
        id: "",
        name: "",
        type: "",
        model_id: "",
        case_sensitive: false,
        ignore_punctuation: false
      });
    }
  }, [mode, config]);
  const handleSubmit = () => {
    onSave(localConfig);
  };

  const mode_str = mode === "new" ? t('evaluation.new') : t('evaluation.modify');
  const titleKey = mode === "new" ? 'evaluation.newEvaluatorConfig' : 'evaluation.editEvaluatorConfig';
  const descKey = mode === "new" ? 'evaluation.newEvaluatorConfigDesc' : 'evaluation.editEvaluatorConfigDesc';

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{t(titleKey)}</DialogTitle>
          <DialogDescription>{t(descKey)}</DialogDescription>
        </DialogHeader>

        <div className="grid gap-6 py-4">
          {/* Name */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label htmlFor="setting_name">{t('evaluation.configName')}</Label>
            <Input
              id="setting_name"
              value={localConfig.name}
              onChange={(e) =>
                setLocalConfig((prev) => ({ ...prev, name: e.target.value }))
              }
              placeholder={t('evaluation.configNamePlaceholder')}
              required
            />
          </div>

          

          {/* Evaluator selection */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4 border-t  pt-3">
            <Label htmlFor="enable_agent">{t('evaluation.evaluatorSelection')}</Label>
            <div className="space-y-4 w-full">
              {/* Evaluator type selection */}
              <Select
                value={localConfig.type || ""}
                onValueChange={(value) => {
                  setLocalConfig((prev) => ({
                    ...prev,
                    type: value,
                  }));
                }}
              >
                <SelectTrigger id="evaluator_name">
                  <SelectValue placeholder={t('evaluation.selectEvaluator')} />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="ExactMatch">{t('evaluation.exactMatch')}</SelectItem>
                  <SelectItem value="LLMJudge">{t('evaluation.llmJudge')}</SelectItem>
                </SelectContent>
              </Select>

              {/* Dynamic configuration area */}
              {localConfig.type === "ExactMatch" && (
                <div className="space-y-3 pt-3">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="case_sensitive" className="text-sm">
                      {t('evaluation.caseSensitive')}
                    </Label>
                    <Switch
                      id="case_sensitive"
                      checked={localConfig.case_sensitive || false}
                      onCheckedChange={(checked) => {
                        setLocalConfig((prev) => ({
                          ...prev,
                          case_sensitive: checked,
                        }));
                      }}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label htmlFor="ignore_punctuation" className="text-sm">
                      {t('evaluation.ignorePunctuation')}
                    </Label>
                    <Switch
                      id="ignore_punctuation"
                      checked={localConfig.ignore_punctuation ?? true}
                      onCheckedChange={(checked) => {
                        setLocalConfig((prev) => ({
                          ...prev,
                          ignore_punctuation: checked,
                        }));
                      }}
                    />
                  </div>
                </div>
              )}

              {localConfig.type === "LLMJudge" && (
                <div className="pt-3">
                  <Label htmlFor="model_id" className="block text-sm mb-2">
                    {t('evaluation.selectEvaluatorModel')}
                  </Label>
                  <Select
                    value={localConfig.model_id || ""}
                    onValueChange={(value) => {
                      setLocalConfig((prev) => ({
                        ...prev,
                        model_id: value,
                      }));
                    }}
                  >
                    <SelectTrigger id="model_id">
                      <SelectValue placeholder={t('evaluation.selectEvalModel')} />
                    </SelectTrigger>
                    <SelectContent>
                      {llms.map((llm) => (
                        <SelectItem key={llm.model_id} value={llm.model_id}>
                          {llm.model_id}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
            </div>
          </div>
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t('common.cancel')}
          </Button>
          <Button onClick={handleSubmit} disabled={isSaving}>
            {isSaving ? (
              <>
                <Spinner size="sm" className="mr-2" />
                {t('evaluation.submitting')}
              </>
            ) : (
              mode_str
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}