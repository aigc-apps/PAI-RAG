import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { MCPConfig } from "@/app/config/mcp/page";
import type { FC } from "react";
import { useState, useEffect } from "react";

interface McpModalProps {
  mcpConfigs: McpEntry[];
  isOpen: boolean;
  onSave: (configs: McpEntry[]) => void;
  onClose: () => void;
  isLoading: boolean;
  error: string | null;
}

export class McpEntry extends MCPConfig {
  active: boolean = false;

  constructor(
    id: string,
    name: string,
    url: string,
    type: string,
    enabled: boolean,
    active: boolean,
  ) {
    super(id, name, url, type, "", false, enabled);
    this.active = active;
  }
}

export const McpModal: FC<McpModalProps> = ({
  mcpConfigs,
  isOpen,
  onSave,
  onClose,
  isLoading,
  error,
}) => {
  const [configs, setConfigs] = useState<McpEntry[]>([]);

  useEffect(() => {
    if (mcpConfigs.length > 0) {
      setConfigs(JSON.parse(JSON.stringify(mcpConfigs))); // 深拷贝
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
    onSave(configs); // 直接保存 configs
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>MCP配置</DialogTitle>
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
                  variant={cfg.active ? "default" : "outline"}
                  onClick={() => toggleActive(cfg.id)}
                >
                  {cfg.active ? "激活" : "激活"}
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
