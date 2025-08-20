// EmbeddingModelDialog.tsx
import { useState, useEffect, FC } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Alert, AlertTitle } from '@/components/ui/alert';
import { AlertCircleIcon } from 'lucide-react';
import { Switch } from '@/components/ui/switch';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';

// 定义组件 props
interface EmbeddingModelDialogProps {
  isAdd: boolean;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  embConfig: EmbConfig;
  onSaveSuccess: (emb: EmbConfig) => void;
}

// 模型数据类型
interface EmbConfig {
  id: string;
  model_id: string;
  model_name: string;
  type: string;
  api_key: string;
  endpoint: string;
  dimension: number;
  embed_batch_size: number;
  is_ready: boolean;
  is_default: boolean;
}

export const EmbeddingModelDialog: FC<EmbeddingModelDialogProps> = ({
  isAdd,
  isOpen,
  setIsOpen,
  embConfig,
  onSaveSuccess,
}) => {
  const [emb, setEmb] = useState<EmbConfig>(embConfig);
  const [error, setError] = useState<string | null>(null);
  const [saveErrorMsg, setSaveErrorMsg] = useState(''); // 保存错误信息

  useEffect(() => {
    setEmb(embConfig);
  }, [isAdd, embConfig]);

  useEffect(() => {
    setSaveErrorMsg('');
  }, [emb]);

  const handleSubmit = async () => {
    setSaveErrorMsg('');
    const is_api_model = emb.type != 'local';
    if (
      is_api_model &&
      (!emb.model_id ||
        (isAdd && !emb.api_key) ||
        !emb.endpoint ||
        !emb.model_name ||
        !emb.type)
    ) {
      setSaveErrorMsg('请必须填写完整的模型信息');
      return;
    } else if (
      !is_api_model &&
      (!emb.model_id || !emb.model_name || !emb.dimension || !emb.type)
    ) {
      setSaveErrorMsg('请必须填写完整的模型信息');
      return;
    }
    const submit_url = isAdd
      ? `${process.env.NEXT_PUBLIC_BACKEND_URL ?? ''}/v1/config/embeddings`
      : `${process.env.NEXT_PUBLIC_BACKEND_URL ?? ''}/v1/config/embeddings/${emb.id}`;
    const updateMethod = isAdd ? 'POST' : 'PATCH';
    if (emb.api_key === '******') emb.api_key = '';
    console.log('updateMethod', isAdd, updateMethod, submit_url, emb);
    try {
      const res = await fetch(submit_url, {
        method: updateMethod,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(emb),
      });

      if (!res.ok) {
        setSaveErrorMsg(`${updateMethod} 请求失败, 请检查填写信息`);
        return;
      }
      const jsondata = await res.json();
      onSaveSuccess(jsondata.data as EmbConfig); // 触发回调
      setIsOpen(false);
    } catch (err: unknown) {
      setSaveErrorMsg(`${updateMethod} 请求失败`);
    }
  };

  // 对话框关闭时重置表单
  const handleDialogClose = (open: boolean) => {
    setIsOpen(open);
    if (!open) {
      setError(null);
      setSaveErrorMsg('');
    }
  };

  // 管理确认对话框的打开状态
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  // 临时存储用户选择的目标状态
  const [pendingState, setPendingState] = useState<boolean | null>(null);

  return (
    <Dialog open={isOpen} onOpenChange={handleDialogClose}>
      <DialogContent className="sm:max-w-[700px]">
        {error && <div className="text-red-500 mb-4">{error}</div>}
        <DialogHeader>
          <DialogTitle>{isAdd ? '添加模型' : '编辑模型'}</DialogTitle>
          <DialogDescription>填写模型配置信息后，点击保存。</DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 py-2">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="model_id" className="text-right">
              模型ID
              <span className="text-destructive">*</span>
            </Label>
            <Input
              id="model_id"
              placeholder="model_id"
              value={emb?.model_id ?? ''}
              onChange={(e) =>
                setEmb((prev) => ({ ...prev, model_id: e.target.value }))
              }
              className="col-span-3"
            />
          </div>
        </div>
        {emb?.type === 'openai_like' && (
          <div className="grid grid-cols-4 items-center gap-4 py-2">
            <Label htmlFor="endpoint" className="text-right">
              Endpoint URL
              <span className="text-destructive">*</span>
            </Label>
            <div className="col-span-3">
              <input
                id="endpoint"
                list="endpoint_options"
                placeholder="输入或选择模型endpoint"
                value={emb?.endpoint ?? ''}
                onChange={(e) =>
                  setEmb((prev) => ({ ...prev, endpoint: e.target.value }))
                }
                className="w-full border border-gray-300 rounded-md p-2 text-sm"
              />
              <datalist id="endpoint_options">
                <option value="https://api.openai.com/v1">OpenAI</option>
                <option value="https://dashscope.aliyuncs.com/compatible-mode/v1">
                  通义千问
                </option>
                {/* 添加更多预设选项 */}
              </datalist>
            </div>
          </div>
        )}
        {emb?.type === 'openai_like' && (
          <div className="grid grid-cols-4 items-center gap-4 py-2">
            <Label htmlFor="api_key" className="text-right">
              API Key
              <span className="text-destructive">*</span>
            </Label>
            {isAdd ? (
              <Input
                id="api_key"
                type="password"
                placeholder="api_key"
                value={emb?.api_key ?? ''}
                onChange={(e) =>
                  setEmb((prev) => ({ ...prev, api_key: e.target.value }))
                }
                className="col-span-3"
              />
            ) : (
              <Input
                id="api_key"
                type="password"
                placeholder="api_key"
                value={emb?.api_key || '******'}
                onChange={(e) =>
                  setEmb((prev) => ({ ...prev, api_key: e.target.value }))
                }
                className="col-span-3"
              />
            )}
          </div>
        )}
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="model_name" className="text-right">
              模型名称
              <span className="text-destructive">*</span>
            </Label>
            <Input
              id="model_name"
              placeholder="model_name"
              value={emb?.model_name ?? ''}
              onChange={(e) =>
                setEmb((prev) => ({ ...prev, model_name: e.target.value }))
              }
              className="col-span-3"
            />
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="model_type" className="text-right">
              模型类型
              <span className="text-destructive">*</span>
            </Label>
            <RadioGroup
              className="flex flex-row gap-6 col-span-3"
              value={emb?.type}
              onValueChange={(value) => setEmb({ ...emb, type: value })}
            >
              <div className="flex items-center gap-3">
                <RadioGroupItem value="local" />
                <Label>本地 (Local Hosted)</Label>
              </div>
              <div className="flex items-center gap-3">
                <RadioGroupItem value="openai_like" />
                <Label>API (OpenAI Like)</Label>
              </div>
            </RadioGroup>
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4 py-2">
            <Label htmlFor="dimension" className="text-right">
              向量维度
            </Label>
            <div className="col-span-3">
              <Input
                id="dimension"
                type="number"
                placeholder="向量维度"
                defaultValue={emb?.dimension || 'null'}
                onChange={(e) =>
                  setEmb({
                    ...emb,
                    dimension: Number(e.target.value),
                  })
                }
              />
            </div>
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4 py-2">
            <Label htmlFor="embed_batch_size" className="text-right">
              向量Batch大小
            </Label>
            <Input
              id="embed_batch_size"
              type="number"
              placeholder="向量Batch大小"
              defaultValue={emb?.embed_batch_size || 'null'}
              onChange={(e) =>
                setEmb({
                  ...emb,
                  embed_batch_size: Number(e.target.value),
                })
              }
              className="col-span-3"
            />
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4 py-2">
            <Label htmlFor="embed_batch_size" className="text-right">
              默认向量模型
            </Label>
            <Switch
              checked={emb.is_default}
              className="justify-start rounded-full transition-color"
              onCheckedChange={(checked) => {
                const targetState = checked;
                if (emb.is_default === targetState) return;
                setPendingState(targetState);
                setIsDialogOpen(true);
              }}
            />
          </div>
        </div>
        {/* 确认对话框 */}
        <AlertDialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>确认更改默认向量模型？</AlertDialogTitle>
              <AlertDialogDescription>
                {pendingState
                  ? '将此模型设为默认后，之前上传的附件都将被清空。'
                  : '取消设为默认后，必须重新指定一个新的默认向量模型。'}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>取消</AlertDialogCancel>
              <AlertDialogAction
                onClick={() => {
                  setEmb({
                    ...emb,
                    is_default: pendingState || false,
                  });
                  setIsDialogOpen(false);
                }}
              >
                确认更改
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        <DialogFooter className="flex flex-col gap-4">
          {saveErrorMsg !== '' && (
            <Alert className="bg-destructive/10 dark:bg-destructive/20 border-none">
              <AlertCircleIcon className="h-4 w-4 !text-destructive" />
              <AlertTitle>{saveErrorMsg}</AlertTitle>
            </Alert>
          )}
          <Button onClick={handleSubmit}>{isAdd ? '新增' : '保存'}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
