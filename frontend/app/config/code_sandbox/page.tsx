'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from 'sonner';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch'; // 确保你有 Switch 组件
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
export default function CodeSandboxConfig() {
  const [isEnabled, setIsEnabled] = useState(false);
  const [configType, setConfigType] = useState('aliyun-fc'); // 目前仅支持 aliyun-fc
  const [aliyunId, setAliyunId] = useState('');
  const [interpreterId, setInterpreterId] = useState('');
  const [interpreterName, setInterpreterName] = useState('');
  const [timeoutDefault, setTimeoutDefault] = useState(50);
  const [isSaving, setIsSaving] = useState(false);
  const { tenantFetch } = useTenantFetch();
  // 初始化加载配置
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const res = await tenantFetch(`/api/config/code_sandbox`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (!res.ok) throw new Error('加载配置失败');

        const response = await res.json();
        const config = response?.data?.[0];

        if (config) {
          setIsEnabled(config.enabled || false);
          setConfigType(config.type || 'aliyun-fc');
          setAliyunId(config.aliyun_id || '');
          setInterpreterId(config.interpreter_id || '');
          setInterpreterName(config.interpreter_name || '');
          setTimeoutDefault(config.timeout_default || 50);
        }
      } catch (err: any) {
        toast.error(err.message);
      }
    };

    fetchConfig();
  }, []);

  const handleSave = async () => {
    try {
      setIsSaving(true);

      const payload: Record<string, any> = {
        enabled: isEnabled,
        type: configType,
        timeout_default: timeoutDefault,
      };

      // 仅当用户输入了值才提交（避免覆盖已有值为空）
      if (aliyunId) payload.aliyun_id = aliyunId;
      if (interpreterId) payload.interpreter_id = interpreterId;
      if (interpreterName) payload.interpreter_name = interpreterName;
      if (isEnabled) payload.enabled = isEnabled;
      if (timeoutDefault) payload.timeout_default = timeoutDefault;


      const res = await tenantFetch(`/api/config/code_sandbox`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error('保存失败，请检查网络或配置');

      toast.success('代码沙箱配置已成功保存。');
    } catch (err: any) {
      toast.warning(err.message);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div id="codesandbox">
      <div className="transition-colors rounded-lg p-4 overflow-hidden duration-200">
        <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-gray-200 rounded-xl bg-gray-50">
          <h2 className="text-xl font-medium text-gray-800">代码沙箱配置</h2>

          <div className="grid gap-4 py-4 w-full max-w-2xl">
            {/* 启用开关 */}
            <div className="grid grid-cols-4 items-center gap-4">
              <Label className="text-right">启用沙箱</Label>
              <div className="col-span-3 flex items-center space-x-2">
                <Switch
                  checked={isEnabled}
                  onCheckedChange={setIsEnabled}
                />
                <span className="text-sm text-gray-600">
                  {isEnabled ? '已启用' : '已禁用'}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-4 items-center gap-4">
              <Label className="text-right">沙箱类型</Label>
              <div className="col-span-3">
                <Select value={configType} onValueChange={setConfigType} disabled>
                  <SelectTrigger>
                    <SelectValue placeholder="请选择沙箱类型" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="aliyun-fc">阿里云FC沙箱</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-gray-500 mt-1">当前仅支持阿里云FC沙箱</p>
              </div>
            </div>

            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="aliyun_id" className="text-right">
                阿里云 ID
              </Label>
              <div className="col-span-3">
                <Input
                  id="aliyun_id"
                  value={aliyunId}
                  onChange={(e) => setAliyunId(e.target.value)}
                  placeholder={'输入阿里云ID'}
                />
              </div>
            </div>

            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="interpreter_id" className="text-right">
                解释器 ID
              </Label>
              <div className="col-span-3">
                <Input
                  id="interpreter_id"
                  value={interpreterId}
                  onChange={(e) => setInterpreterId(e.target.value)}
                  placeholder={'输入code解释器 ID'}
                />
              </div>
            </div>

            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="interpreter_name" className="text-right">
                解释器名称
              </Label>
              <div className="col-span-3">
                <Input
                  id="interpreter_name"
                  value={interpreterName}
                  onChange={(e) => setInterpreterName(e.target.value)}
                  placeholder={'输入解释器名称（templateName）'}
                />
              </div>
            </div>

            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="timeout_default" className="text-right">
                默认超时（秒）({timeoutDefault})
              </Label>
              <div className="col-span-3">
                <Slider
                  value={[timeoutDefault]}
                  max={300}
                  min={10}
                  step={5}
                  onValueChange={(value) => setTimeoutDefault(value[0])}
                />
              </div>
            </div>
          </div>

          <Button
            onClick={handleSave}
            disabled={isSaving}
            className="mt-4 px-4 py-2 text-white rounded-lg transition-colors"
          >
            {isSaving ? '保存中...' : '保存沙箱配置'}
          </Button>
        </div>
      </div>
    </div>
  );
}