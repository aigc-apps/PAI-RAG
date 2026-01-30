'use client';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { McpConfig } from '@/app/config/mcp/mcp';
import type { FC } from 'react';
import { useState, useEffect } from 'react';
import { useChatOptions } from '@/app/providers/chat';
import { useI18n } from '@/app/providers/i18n';

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
  const {mcp_ids, updateMcpIds} = useChatOptions();

  useEffect(() => {
    if (mcpConfigs.length > 0) {
      setConfigs(JSON.parse(JSON.stringify(mcpConfigs))); // Deep copy
    }
  }, [mcpConfigs]);

  const toggleActive = (id: string) => {
    setConfigs((prev) => {
      const updated = prev.map((cfg) => {
        if (cfg.id === id) {
          return { ...cfg, active: !cfg.active };
        }
        return cfg;
      });
      return updated;
    });
  };

  const handleSave = () => {
    onSave(configs); // Save configs directly
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t('mcp.modalTitle')}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          {isLoading ? (
            <p>{t('common.loading')}</p>
          ) : error ? (
            <p className="text-red-500">{error}</p>
          ) : (
            configs.map((cfg) => (
              <div key={cfg.id} className="flex justify-between items-center">
                <span>{cfg.name}</span>
                <Button
                  variant={cfg.active ? 'default' : 'outline'}
                  onClick={() => toggleActive(cfg.id)}
                >
                  {cfg.active ? t('mcp.activate') : t('mcp.activate')}
                </Button>
              </div>
            ))
          )}
        </div>
        <Button onClick={handleSave}>{t('common.save')}</Button>
      </DialogContent>
    </Dialog>
  );
};
