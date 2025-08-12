import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogClose,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { KnowledgeBase } from '@/app/knowledgebases/page';
import type { FC } from 'react';
import { useState, useEffect } from 'react';

export class KbSelection implements KnowledgeBase {
  id: string;
  name: string;
  description: string;
  active: boolean;
  updated_at: string;

  constructor(id: string, name: string, description: string, active: boolean, updated_at: string) {
    this.id = id;
    this.name = name;
    this.description = description;
    this.active = active;
    this.updated_at = updated_at;
  }
}

interface KbModalProps {
  kbConfigs: KbSelection[];
  isOpen: boolean;
  onSave: (configs: KbSelection[]) => void;
  onClose: () => void;
  isLoading: boolean;
  error: string | null;
}

export const KbModal: FC<KbModalProps> = ({
  kbConfigs,
  isOpen,
  onSave,
  onClose,
  isLoading,
  error,
}) => {
  const [configs, setConfigs] = useState<KbSelection[]>([]);

  useEffect(() => {
    if (kbConfigs.length > 0) {
      setConfigs(kbConfigs); // 深拷贝
    }
  }, [kbConfigs]);

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
    onSave(configs); // 直接保存 configs
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>知识库配置</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          {isLoading ? (
            <p>加载中...</p>
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
                  {cfg.active ? '激活' : '激活'}
                </Button>
              </div>
            ))
          )}
        </div>
        <Button onClick={handleSave}>保存</Button>
      </DialogContent>
    </Dialog>
  );
};
