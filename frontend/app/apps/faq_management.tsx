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
import { Plus, Edit, Trash2, Settings, HelpCircle, Upload } from 'lucide-react';
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
  const [isUploadDialogOpen, setIsUploadDialogOpen] = useState(false);
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadConfig, setUploadConfig] = useState<{
    header_index_max: number | null;
    question_column_index: number;
    answer_column_index: number;
  }>({
    header_index_max: 0,
    question_column_index: 0,
    answer_column_index: 1,
  });
  const [uploading, setUploading] = useState(false);
  const [editingFaq, setEditingFaq] = useState<FAQItem | null>(null);
  const [formData, setFormData] = useState<FAQItem>({ question: '', answer: '' });
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModel[]>([]);
  const [faqConfigData, setFaqConfigData] = useState<{
    active: boolean;
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
      // 更新 faq_config.active 字段
      const res = await tenantFetch(`/api/config/apps/${appId}/faq-config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          active: checked,
        }),
      });

      if (!res.ok) throw new Error('更新失败');
      
      const data = await res.json();
      // 更新本地状态
      if (data.data) {
        setFaqConfigData({
          active: data.data.active ?? checked,
          score_threshold: data.data.similarity_threshold ?? data.data.score_threshold ?? faqConfigData?.score_threshold ?? 0.9,
          embedding_model: data.data.embedding_model ?? faqConfigData?.embedding_model ?? '',
          question_in_retrieval: data.data.question_in_retrieval ?? faqConfigData?.question_in_retrieval ?? true,
          question_in_response: data.data.question_in_response ?? faqConfigData?.question_in_response ?? false,
          answer_in_retrieval: data.data.answer_in_retrieval ?? faqConfigData?.answer_in_retrieval ?? false,
          answer_in_response: data.data.answer_in_response ?? faqConfigData?.answer_in_response ?? true,
        });
      }
      
      toast.success(checked ? '已启用FAQ回复' : '已关闭FAQ回复');
    } catch (error: any) {
      toast.error(error.message || '更新失败');
    }
  };

  useEffect(() => {
    fetchFAQs();
    fetchEmbeddingModels();
    // 无论 enable_faq 是否为 true，都加载 FAQ 配置以获取 active 状态
    fetchFAQConfig();
  }, [appId]);

  const fetchFAQConfig = async () => {
    try {
      const res = await tenantFetch(`/api/config/apps/${appId}/faq-config`);
      if (res.ok) {
        const data = await res.json();
        if (data.data) {
          // 从后端返回的数据中提取配置字段
          setFaqConfigData({
            active: data.data.active ?? false,
            score_threshold: data.data.score_threshold ?? data.data.similarity_threshold ?? 0.9,
            embedding_model: data.data.embedding_model ?? '',
            question_in_retrieval: data.data.question_in_retrieval ?? true,
            question_in_response: data.data.question_in_response ?? false,
            answer_in_retrieval: data.data.answer_in_retrieval ?? false,
            answer_in_response: data.data.answer_in_response ?? true,
          });
        } else {
          // 如果没有配置数据，设置默认值（active 默认为 false）
          setFaqConfigData({
            active: false,
            score_threshold: 0.9,
            embedding_model: '',
            question_in_retrieval: true,
            question_in_response: false,
            answer_in_retrieval: false,
            answer_in_response: true,
          });
        }
      } else if (res.status === 404) {
        // FAQ 配置不存在，设置默认值
        setFaqConfigData({
          active: false,
          score_threshold: 0.9,
          embedding_model: '',
          question_in_retrieval: true,
          question_in_response: false,
          answer_in_retrieval: false,
          answer_in_response: true,
        });
      }
    } catch (error: any) {
      console.error('获取FAQ配置失败:', error);
      // 即使出错也设置默认值，确保开关可以显示
      setFaqConfigData({
        active: false,
        score_threshold: 0.9,
        embedding_model: '',
        question_in_retrieval: true,
        question_in_response: false,
        answer_in_retrieval: false,
        answer_in_response: true,
      });
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

  const handleUploadFiles = async () => {
    if (uploadFiles.length === 0) {
      toast.error('请选择要上传的文件');
      return;
    }

    // 验证文件类型
    const validFiles = uploadFiles.filter(
      (file) =>
        file.name.endsWith('.xlsx') || file.name.endsWith('.xls')
    );

    if (validFiles.length === 0) {
      toast.error('请选择有效的Excel文件（.xlsx 或 .xls）');
      return;
    }

    setUploading(true);
    try {
      // 构建 table_config (扁平结构，不嵌套 faq_config)
      const tableConfig = {
        header_index_max: uploadConfig.header_index_max,
        question_column_index: uploadConfig.question_column_index,
        answer_column_index: uploadConfig.answer_column_index,
      };

      // 创建 FormData
      const formData = new FormData();
      validFiles.forEach((file) => {
        formData.append('files', file);
      });
      formData.append('table_config', JSON.stringify(tableConfig));

      const res = await tenantFetch(`/api/config/apps/${appId}/faq-files`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.message || '上传失败');
      }

      const data = await res.json();
      const successCount = data.data?.filter(
        (item: any) => item.chunks_count > 0
      ).length || 0;
      const totalChunks = data.data?.reduce(
        (sum: number, item: any) => sum + (item.chunks_count || 0),
        0
      ) || 0;

      toast.success(
        `成功上传 ${successCount}/${validFiles.length} 个文件，共提取 ${totalChunks} 个片段`
      );

      // 关闭对话框并重置状态
      setIsUploadDialogOpen(false);
      setUploadFiles([]);
      setUploadConfig({
        header_index_max: 0,
        question_column_index: 0,
        answer_column_index: 1,
      });

      // 刷新FAQ列表
      fetchFAQs();
    } catch (error: any) {
      toast.error(error.message || '上传失败');
    } finally {
      setUploading(false);
    }
  };

  const handleSaveConfig = async () => {
    try {
      const res = await tenantFetch(`/api/config/apps/${appId}/faq-config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          active: faqConfigData?.active ?? true,
          similarity_threshold: faqConfigData?.score_threshold,
          embedding_model: faqConfigData?.embedding_model,
          question_in_retrieval: faqConfigData?.question_in_retrieval,
          question_in_response: faqConfigData?.question_in_response,
          answer_in_retrieval: faqConfigData?.answer_in_retrieval,
          answer_in_response: faqConfigData?.answer_in_response,
        }),
      });

      if (!res.ok) throw new Error('保存配置失败');
      
      const data = await res.json();
      // 更新本地状态，确保包含 active 字段
      if (data.data && faqConfigData) {
        setFaqConfigData({
          ...faqConfigData,
          active: data.data.active ?? faqConfigData.active,
        });
      }
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
            checked={faqConfigData?.active ?? false}
            onCheckedChange={handleToggleFAQ}
          />
        </div>
        {botConfig.enable_faq && faqConfigData && (
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
        <Button onClick={() => setIsUploadDialogOpen(true)} size="sm" variant="outline">
          <Upload className="w-4 h-4 mr-2" />
          上传文件
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
                      min={0}
                      max={1}
                      step={0.01}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>0 · 容易匹配</span>
                      <span className="font-medium">{(faqConfigData?.score_threshold ?? 0.9).toFixed(2)}</span>
                      <span>1 · 精准匹配</span>
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

      {/* 上传文件对话框 */}
      <Dialog open={isUploadDialogOpen} onOpenChange={setIsUploadDialogOpen}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle>上传文件</DialogTitle>
            <DialogDescription>
              选择文件进行上传
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            {/* 文件选择 */}
            <div className="space-y-2">
              <Label htmlFor="file-upload">选择文件</Label>
              <div className="border-2 border-dashed border-muted rounded-lg p-6 text-center hover:border-primary/50 transition-colors">
                <input
                  id="file-upload"
                  type="file"
                  accept=".xlsx,.xls"
                  multiple
                  onChange={(e) => {
                    const files = Array.from(e.target.files || []);
                    setUploadFiles(files);
                  }}
                  className="hidden"
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer flex flex-col items-center gap-2"
                >
                  <Upload className="w-8 h-8 text-muted-foreground" />
                  <span className="text-sm font-medium">点击选择文件</span>
                  <span className="text-xs text-muted-foreground">支持的文件类型: xlsx, xls</span>
                </label>
                {uploadFiles.length > 0 && (
                  <div className="mt-4 space-y-2 text-left">
                    <div className="text-xs text-muted-foreground mb-2">已选择 {uploadFiles.length} 个文件:</div>
                    {uploadFiles.map((file, index) => (
                      <div key={index} className="text-sm text-foreground bg-muted/50 rounded px-2 py-1">
                        {file.name}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* 配置项 */}
            <div className="space-y-4 border-t pt-4">
              <Label>文件解析配置</Label>
              
              {/* 标题行下标 */}
              <div className="space-y-2">
                <Label htmlFor="header_index_max" className="text-sm">
                  标题行下标
                </Label>
                <Input
                  id="header_index_max"
                  type="number"
                  min="0"
                  value={uploadConfig.header_index_max ?? ''}
                  onChange={(e) =>
                    setUploadConfig({
                      ...uploadConfig,
                      header_index_max: e.target.value === '' ? null : parseInt(e.target.value) || 0,
                    })
                  }
                  placeholder="留空表示不使用标题行，默认: 0"
                />
                <p className="text-xs text-muted-foreground">
                  留空表示不使用任何行作为标题行，列将使用数字索引（0, 1, 2...）
                </p>
              </div>

              {/* 问题列 */}
              <div className="space-y-2">
                <Label htmlFor="question_column_index" className="text-sm">
                  问题列
                </Label>
                <Input
                  id="question_column_index"
                  type="number"
                  min="0"
                  value={uploadConfig.question_column_index}
                  onChange={(e) =>
                    setUploadConfig({
                      ...uploadConfig,
                      question_column_index: parseInt(e.target.value) || 0,
                    })
                  }
                  placeholder="默认: 0"
                />
              </div>

              {/* 答案列 */}
              <div className="space-y-2">
                <Label htmlFor="answer_column_index" className="text-sm">
                  答案列
                </Label>
                <Input
                  id="answer_column_index"
                  type="number"
                  min="0"
                  value={uploadConfig.answer_column_index}
                  onChange={(e) =>
                    setUploadConfig({
                      ...uploadConfig,
                      answer_column_index: parseInt(e.target.value) || 1,
                    })
                  }
                  placeholder="默认: 1"
                />
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setIsUploadDialogOpen(false);
                setUploadFiles([]);
                setUploadConfig({
                  header_index_max: 0,
                  question_column_index: 0,
                  answer_column_index: 1,
                });
              }}
            >
              取消
            </Button>
            <Button
              onClick={handleUploadFiles}
              disabled={uploadFiles.length === 0 || uploading}
            >
              {uploading ? '上传中...' : '上传'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

