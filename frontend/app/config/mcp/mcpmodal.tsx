'use client';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { McpConfig } from '@/app/config/mcp/mcp';
import type { FC } from 'react';
import { useState, useEffect } from 'react';
import { useChatOptions } from '@/app/providers/chat';
import { useI18n } from '@/app/providers/i18n';
import { PlugZap, Check, Loader2 } from 'lucide-react';

export class McpEntry extends McpConfig {
  active: boolean = false;

  constructor(
    id: string,
    name: string,
    url: string,
    type: string,
    enabled: boolean,
    active: boolean,
  ) {
    super(id, name, url, type, '', false, enabled);
    this.active = active;
  }
}

interface McpModalProps {
  mcpConfigs: McpEntry[];
  isOpen: boolean;
  onSave: (configs: McpEntry[]) => void;
  onClose: () => void;
  isLoading: boolean;
  error: string | null;
}

export const McpModal: FC<McpModalProps> = ({
  mcpConfigs,
  isOpen,
  onSave,
  onClose,
  isLoading,
  error,
}) => {
  const { t } = useI18n();
  const [configs, setConfigs] = useState<McpEntry[]>([]);
  const { mcp_ids, updateMcpIds } = useChatOptions();

  useEffect(() => {
    if (mcpConfigs.length > 0) {
      setConfigs(JSON.parse(JSON.stringify(mcpConfigs)));
    }
  }, [mcpConfigs]);

  const toggleActive = (id: string) => {
    setConfigs((prev) =>
      prev.map((cfg) => (cfg.id === id ? { ...cfg, active: !cfg.active } : cfg)),
    );
  };

  const handleSave = () => onSave(configs);
  const activeCount = configs.filter((c) => c.active).length;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md gap-0 p-0 overflow-hidden">
        <DialogHeader className="px-5 pt-5 pb-3 border-b border-border">
          <DialogTitle className="flex items-center gap-2 text-sm font-semibold">
            <PlugZap className="w-4 h-4 text-primary" />
            {t('mcp.modalTitle')}
          </DialogTitle>
          <DialogDescription className="text-xs mt-0.5">
            MCP
            {activeCount > 0 && (
              <span className="ml-1 text-primary font-medium">
                · {activeCount} {t('common.selected') || '已选'}
              </span>
            )}
          </DialogDescription>
        </DialogHeader>

        <div className="max-h-[360px] overflow-y-auto p-2 space-y-0.5">
          {isLoading ? (
            <div className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              {t('common.loading')}
            </div>
          ) : error ? (
            <p className="text-xs text-destructive text-center py-6">{error}</p>
          ) : configs.length === 0 ? (
            <div className="text-center py-8 text-xs text-muted-foreground">
              <PlugZap className="w-8 h-8 mx-auto mb-2 opacity-40" />
              {t('common.noData') || '暂无数据'}
            </div>
          ) : (
            configs.map((cfg) => {
              const initial = (cfg.name || 'M').charAt(0).toUpperCase();
              return (
                <button
                  key={cfg.id}
                  type="button"
                  onClick={() => toggleActive(cfg.id)}
                  className={`w-full flex items-center gap-2.5 px-2 py-2 rounded-md text-left transition-colors ${
                    cfg.active
                      ? 'bg-primary/10 hover:bg-primary/15'
                      : 'hover:bg-muted/60'
                  }`}
                >
                  <div
                    className={`model-icon type-llm shrink-0 !w-7 !h-7 !text-[11px] ${
                      cfg.active ? '' : 'opacity-80'
                    }`}
                  >
                    {initial}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div
                      className={`text-xs font-medium truncate ${
                        cfg.active ? 'text-primary' : 'text-foreground'
                      }`}
                    >
                      {cfg.name}
                    </div>
                    {cfg.url && (
                      <div className="text-[10px] font-mono text-muted-foreground truncate mt-0.5">
                        {cfg.url}
                      </div>
                    )}
                  </div>
                  <div
                    className={`w-4 h-4 rounded-full flex items-center justify-center shrink-0 transition-colors ${
                      cfg.active
                        ? 'bg-primary text-primary-foreground'
                        : 'border border-border bg-background'
                    }`}
                  >
                    {cfg.active && <Check className="w-3 h-3" strokeWidth={3} />}
                  </div>
                </button>
              );
            })
          )}
        </div>

        <DialogFooter className="px-5 py-3 border-t border-border bg-muted/20">
          <Button variant="outline" size="sm" onClick={onClose}>
            {t('common.cancel')}
          </Button>
          <Button size="sm" onClick={handleSave}>
            {t('common.save')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
