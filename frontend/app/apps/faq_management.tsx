'use client';
import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Plus, Edit, Trash2, Settings, HelpCircle } from 'lucide-react';
import { toast } from 'sonner';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { Switch } from '@/components/ui/switch';
import { Chatbot } from './chatbot_config';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

interface FAQItem {
  id?: string;
  question: string;
  answer: string;
}

interface EmbeddingModel {
  id: string;
  model_id: string;
  model_name: string;
  type: string;
  provider_name?: string;
}

interface FAQManagementProps {
  appId: string;
  botConfig: Chatbot;
  setBotConfig: (config: Chatbot) => void;
}

export const FAQManagement: React.FC<FAQManagementProps> = ({ appId, botConfig, setBotConfig }) => {
  const [faqs, setFaqs] = useState<FAQItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isConfigDialogOpen, setIsConfigDialogOpen] = useState(false);
  const [editingFaq, setEditingFaq] = useState<FAQItem | null>(null);
  const [formData, setFormData] = useState<FAQItem>({ question: '', answer: '' });
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModel[]>([]);
  const [faqConfigData, setFaqConfigData] = useState<{
    score_threshold: number;
    embedding_model: string;
    question_in_retrieval: boolean;
    question_in_response: boolean;
    answer_in_retrieval: boolean;
    answer_in_response: boolean;
  } | null>(null);
  const { tenantFetch } = useTenantFetch();

  const handleToggleFAQ = async (checked: boolean) => {
    try {
      if (checked) {
        // 开启FAQ：创建或获取FAQ配置，设置faq_id
        const configResponse = await tenantFetch(`/api/config/apps/${botConfig.app_id}/faq-config`);
        if (!configResponse.ok) throw new Error('获取FAQ配置失败');
        const configData = await configResponse.json();
        const faqConfigId = configData.data.id;

        const res = await tenantFetch(`/api/config/apps/${botConfig.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ...botConfig,
            faq_id: faqConfigId,
          }),
        });

        if (!res.ok) throw new Error('更新失败');
        
        setBotConfig({ ...botConfig, faq_id: faqConfigId });
        // 加载FAQ配置数据
        if (configData.data) {
          setFaqConfigData({
            score_threshold: configData.data.score_threshold ?? 0.9,
            embedding_model: configData.data.embedding_model ?? '',
            question_in_retrieval: configData.data.question_in_retrieval ?? true,
            question_in_response: configData.data.question_in_response ?? false,
            answer_in_retrieval: configData.data.answer_in_retrieval ?? false,
            answer_in_response: configData.data.answer_in_response ?? true,
          });
        }
        toast.success('已启用FAQ回复');
      } else {
        // 关闭FAQ：清空faq_id
        const res = await tenantFetch(`/api/config/apps/${botConfig.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ...botConfig,
            faq_id: null,
          }),
        });

        if (!res.ok) throw new Error('更新失败');
        
        setBotConfig({ ...botConfig, faq_id: null });
        setFaqConfigData(null);
        toast.success('已关闭FAQ回复');
      }
    } catch (error: any) {
      toast.error(error.message || '更新失败');
    }
  };

  useEffect(() => {
    fetchFAQs();
    fetchEmbeddingModels();
    if (botConfig.faq_id) {
      fetchFAQConfig();
    }
  }, [appId, botConfig.faq_id]);

  const fetchFAQConfig = async () => {
    try {
      const res = await tenantFetch(`/api/config/apps/${botConfig.app_id}/faq-config`);
      if (res.ok) {
        const data = await res.json();
        if (data.data) {
          // 从后端返回的数据中提取配置字段
          setFaqConfigData({
            score_threshold: data.data.score_threshold ?? 0.9,
            embedding_model: data.data.embedding_model ?? '',
            question_in_retrieval: data.data.question_in_retrieval ?? true,
            question_in_response: data.data.question_in_response ?? false,
            answer_in_retrieval: data.data.answer_in_retrieval ?? false,
            answer_in_response: data.data.answer_in_response ?? true,
          });
        } else {
          // 初始化默认配置（与后端一致）
          setFaqConfigData({
            score_threshold: 0.9,
            embedding_model: '',
            question_in_retrieval: true,
            question_in_response: false,
            answer_in_retrieval: false,
            answer_in_response: true,
          });
        }
      }
    } catch (error: any) {
      console.error('获取FAQ配置失败:', error);
    }
  };

  const fetchEmbeddingModels = async () => {
    try {
      const res = await tenantFetch(`/api/config/embeddings?size=1000`);
      if (res.ok) {
        const data = await res.json();
        setEmbeddingModels(data.data?.items || []);
      }
    } catch (error: any) {
      console.error('获取Embedding模型列表失败:', error);
    }
  };

  const fetchFAQs = async () => {
    try {
      setLoading(true);
      const res = await tenantFetch(`/api/config/apps/${appId}/faqs`);
      if (res.ok) {
        const data = await res.json();
        setFaqs(data.data?.items || []);
      }
    } catch (error: any) {
      console.error('获取FAQ列表失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenDialog = (faq?: FAQItem) => {
    if (faq) {
      setEditingFaq(faq);
      setFormData({ question: faq.question, answer: faq.answer });
    } else {
      setEditingFaq(null);
      setFormData({ question: '', answer: '' });
    }
    setIsDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setIsDialogOpen(false);
    setEditingFaq(null);
    setFormData({ question: '', answer: '' });
  };

  const handleSave = async () => {
    if (!formData.question.trim() || !formData.answer.trim()) {
      toast.error('请填写问题和答案');
      return;
    }

    try {
      const url = editingFaq?.id
        ? `/api/config/apps/${appId}/faqs/${editingFaq.id}`
        : `/api/config/apps/${appId}/faqs`;
      const method = editingFaq?.id ? 'PUT' : 'POST';

      const res = await tenantFetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!res.ok) throw new Error('保存失败');
      
      toast.success(editingFaq ? '更新成功' : '创建成功');
      handleCloseDialog();
      fetchFAQs();
    } catch (error: any) {
      toast.error(error.message || '保存失败');
    }
  };

  const handleDelete = async (faqId: string) => {
    if (!confirm('确定要删除这条FAQ吗？')) return;

    try {
      const res = await tenantFetch(`/api/config/apps/${appId}/faqs/${faqId}`, {
        method: 'DELETE',
      });

      if (!res.ok) throw new Error('删除失败');
      
      toast.success('删除成功');
      fetchFAQs();
    } catch (error: any) {
      toast.error(error.message || '删除失败');
    }
  };

  const handleSaveConfig = async () => {
    try {
      const res = await tenantFetch(`/api/config/apps/${botConfig.app_id}/faq-config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          score_threshold: faqConfigData?.score_threshold,
          embedding_model: faqConfigData?.embedding_model,
          question_in_retrieval: faqConfigData?.question_in_retrieval,
          question_in_response: faqConfigData?.question_in_response,
          answer_in_retrieval: faqConfigData?.answer_in_retrieval,
          answer_in_response: faqConfigData?.answer_in_response,
        }),
      });

      if (!res.ok) throw new Error('保存配置失败');
      
      setFaqConfigData(faqConfigData);
      setIsConfigDialogOpen(false);
      toast.success('配置保存成功');
    } catch (error: any) {
      toast.error(error.message || '保存配置失败');
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-end items-center gap-4">
        <div className="flex items-center gap-2">
          <Label htmlFor="enable_faq_switch">开启FAQ回复</Label>
          <Switch
            id="enable_faq_switch"
            checked={!!botConfig.faq_id}
            onCheckedChange={handleToggleFAQ}
          />
        </div>
        {botConfig.faq_id && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              if (!faqConfigData) {
                fetchFAQConfig();
              }
              setIsConfigDialogOpen(true);
            }}
          >
            <Settings className="w-4 h-4 mr-2" />
            配置
          </Button>
        )}
        <Button onClick={() => handleOpenDialog()} size="sm">
          <Plus className="w-4 h-4 mr-2" />
          新增FAQ
        </Button>
      </div>

      {loading ? (
        <div className="text-center py-8 text-muted-foreground">加载中...</div>
      ) : faqs.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground">
          暂无FAQ，点击"新增FAQ"添加
        </div>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[200px]">问题</TableHead>
              <TableHead>答案</TableHead>
              <TableHead className="w-[120px]">操作</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {faqs.map((faq) => (
              <TableRow key={faq.id}>
                <TableCell className="font-medium">{faq.question}</TableCell>
                <TableCell className="max-w-md truncate">{faq.answer}</TableCell>
                <TableCell>
                  <div className="flex gap-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleOpenDialog(faq)}
                    >
                      <Edit className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => faq.id && handleDelete(faq.id)}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}

      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle>{editingFaq ? '编辑FAQ' : '新增FAQ'}</DialogTitle>
            <DialogDescription>
              填写问题和答案，创建FAQ条目
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="question">问题 *</Label>
              <Input
                id="question"
                value={formData.question}
                onChange={(e) =>
                  setFormData({ ...formData, question: e.target.value })
                }
                placeholder="请输入问题"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="answer">答案 *</Label>
              <Textarea
                id="answer"
                value={formData.answer}
                onChange={(e) =>
                  setFormData({ ...formData, answer: e.target.value })
                }
                placeholder="请输入答案"
                rows={6}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={handleCloseDialog}>
              取消
            </Button>
            <Button onClick={handleSave}>
              {editingFaq ? '更新' : '创建'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* FAQ配置对话框 */}
      <Dialog open={isConfigDialogOpen} onOpenChange={setIsConfigDialogOpen}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle>FAQ回复设置</DialogTitle>
            <DialogDescription>
              配置FAQ检索和回复的相关参数
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-6 py-4">
            {!faqConfigData ? (
              <div className="text-center py-4 text-muted-foreground">加载配置中...</div>
            ) : (
              <>
                {/* 分数阈值 */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="score_threshold">分数阈值</Label>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="text-xs">设置FAQ匹配的相似度阈值，值越高匹配越精准</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  <div className="space-y-2">
                    <Slider
                      value={[faqConfigData?.score_threshold ?? 0.9]}
                      onValueChange={(value) =>
                        setFaqConfigData({ ...faqConfigData!, score_threshold: value[0] })
                      }
                      min={0.8}
                      max={1.0}
                      step={0.01}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>0.8 · 容易匹配</span>
                      <span className="font-medium">{(faqConfigData?.score_threshold ?? 0.9).toFixed(2)}</span>
                      <span>1.0 · 精准匹配</span>
                    </div>
                  </div>
                </div>

                {/* Embedding模型 */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="embedding_model">Embedding 模型</Label>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="text-xs">选择用于FAQ向量化的Embedding模型</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  <Select
                    value={faqConfigData?.embedding_model ?? ''}
                    onValueChange={(value) =>
                      setFaqConfigData({ ...faqConfigData!, embedding_model: value })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="请选择Embedding模型" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        {embeddingModels.map((model) => (
                          <SelectItem key={model.id} value={model.model_id}>
                            {model.model_id}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </div>

                {/* 问题是否参与检索/回答 */}
                <div className="space-y-3">
                  <Label>问题参与设置</Label>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="question_in_retrieval" className="text-sm font-normal">
                        问题参与检索
                      </Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-xs">是否使用问题内容进行向量检索</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                    <Switch
                      id="question_in_retrieval"
                      checked={faqConfigData?.question_in_retrieval ?? true}
                      onCheckedChange={(checked) =>
                        setFaqConfigData({ ...faqConfigData!, question_in_retrieval: checked })
                      }
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="question_in_response" className="text-sm font-normal">
                        问题参与回答
                      </Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-xs">是否在回答中包含问题内容</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                    <Switch
                      id="question_in_response"
                      checked={faqConfigData?.question_in_response ?? false}
                      onCheckedChange={(checked) =>
                        setFaqConfigData({ ...faqConfigData!, question_in_response: checked })
                      }
                    />
                  </div>
                </div>

                {/* 答案是否参与检索/回答 */}
                <div className="space-y-3">
                  <Label>答案参与设置</Label>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="answer_in_retrieval" className="text-sm font-normal">
                        答案参与检索
                      </Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-xs">是否使用答案内容进行向量检索</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                    <Switch
                      id="answer_in_retrieval"
                      checked={faqConfigData?.answer_in_retrieval ?? false}
                      onCheckedChange={(checked) =>
                        setFaqConfigData({ ...faqConfigData!, answer_in_retrieval: checked })
                      }
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="answer_in_response" className="text-sm font-normal">
                        答案参与回答
                      </Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-xs">是否在回答中包含答案内容</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                    <Switch
                      id="answer_in_response"
                      checked={faqConfigData?.answer_in_response ?? true}
                      onCheckedChange={(checked) =>
                        setFaqConfigData({ ...faqConfigData!, answer_in_response: checked })
                      }
                    />
                  </div>
                </div>
              </>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsConfigDialogOpen(false)}>
              取消
            </Button>
            <Button onClick={handleSaveConfig}>
              保存
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

