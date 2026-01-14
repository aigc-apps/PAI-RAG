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
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { Checkbox } from '@/components/ui/checkbox';

const DEFAULT_SCORE_THRESHOLD = 0.8;

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
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalItems, setTotalItems] = useState(0);
  const pageSize = 10;
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
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
    enable_question_in_retrieval: boolean;
    enable_question_in_response: boolean;
    enable_answer_in_retrieval: boolean;
    enable_answer_in_response: boolean;
    return_direct: boolean;
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
            score_threshold: data.data.similarity_threshold ?? data.data.score_threshold ?? faqConfigData?.score_threshold ?? DEFAULT_SCORE_THRESHOLD,
            embedding_model: data.data.embedding_model ?? faqConfigData?.embedding_model ?? '',
            enable_question_in_retrieval: data.data.enable_question_in_retrieval ?? faqConfigData?.enable_question_in_retrieval ?? true,
            enable_question_in_response: data.data.enable_question_in_response ?? faqConfigData?.enable_question_in_response ?? false,
            enable_answer_in_retrieval: data.data.enable_answer_in_retrieval ?? faqConfigData?.enable_answer_in_retrieval ?? false,
            enable_answer_in_response: data.data.enable_answer_in_response ?? faqConfigData?.enable_answer_in_response ?? true,
            return_direct: data.data.return_direct ?? faqConfigData?.return_direct ?? false,
          });
      }
      
      // 同步更新 botConfig.enable_faq
      setBotConfig({
        ...botConfig,
        enable_faq: checked,
      });
      
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
  }, [appId, page]);

  // 当页面切换时，清空选中项
  useEffect(() => {
    setSelectedItems(new Set());
  }, [page]);

  const fetchFAQConfig = async () => {
    try {
      const res = await tenantFetch(`/api/config/apps/${appId}/faq-config`);
      if (res.ok) {
        const data = await res.json();
        if (data.data) {
          // 从后端返回的数据中提取配置字段
          setFaqConfigData({
            active: data.data.active ?? false,
            score_threshold: data.data.score_threshold ?? data.data.similarity_threshold ?? DEFAULT_SCORE_THRESHOLD,
            embedding_model: data.data.embedding_model ?? '',
            enable_question_in_retrieval: data.data.enable_question_in_retrieval ?? true,
            enable_question_in_response: data.data.enable_question_in_response ?? true,
            enable_answer_in_retrieval: data.data.enable_answer_in_retrieval ?? false,
            enable_answer_in_response: data.data.enable_answer_in_response ?? true,
            return_direct: data.data.return_direct ?? false,
          });
        } else {
          // 如果没有配置数据，设置默认值（active 默认为 false）
          setFaqConfigData({
            active: false,
            score_threshold: DEFAULT_SCORE_THRESHOLD,
            embedding_model: '',
            enable_question_in_retrieval: true,
            enable_question_in_response: true,
            enable_answer_in_retrieval: false,
            enable_answer_in_response: true,
            return_direct: false,
          });
        }
      } else if (res.status === 404) {
        // FAQ 配置不存在，设置默认值
        setFaqConfigData({
          active: false,
          score_threshold: DEFAULT_SCORE_THRESHOLD,
          embedding_model: '',
          enable_question_in_retrieval: true,
          enable_question_in_response: false,
          enable_answer_in_retrieval: false,
          enable_answer_in_response: true,
          return_direct: false,
        });
      }
    } catch (error: any) {
      console.error('获取FAQ配置失败:', error);
      // 即使出错也设置默认值，确保开关可以显示
      setFaqConfigData({
        active: false,
        score_threshold: DEFAULT_SCORE_THRESHOLD,
        embedding_model: '',
        enable_question_in_retrieval: true,
        enable_question_in_response: false,
        enable_answer_in_retrieval: false,
        enable_answer_in_response: true,
        return_direct: false,
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
      const res = await tenantFetch(`/api/config/apps/${appId}/faqs?page=${page}&size=${pageSize}`);
      if (res.ok) {
        const data = await res.json();
        console.log('FAQ分页数据:', data);
        const items = data.data?.items || [];
        const total = data.data?.total || 0;
        const pages = data.data?.pages || 1;
        
        setFaqs(items);
        setTotalItems(total);
        setTotalPages(pages);
        
        console.log(`FAQ分页信息: 当前页=${page}, 总页数=${pages}, 总条数=${total}, 当前页数据=${items.length}条`);
      } else {
        console.error('获取FAQ列表失败:', res.status, res.statusText);
      }
    } catch (error: any) {
      console.error('获取FAQ列表失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
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
      // 如果是新增，跳转到第一页；如果是更新，保持在当前页
      if (!editingFaq) {
        // 如果当前不在第一页，跳转到第一页（会触发useEffect刷新）
        // 如果已经在第一页，直接刷新列表
        if (page !== 1) {
          setPage(1);
        } else {
          fetchFAQs();
        }
      } else {
        // 更新时刷新当前页列表
        fetchFAQs();
      }
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
      
      // 从选中项中移除
      setSelectedItems(prev => {
        const newSet = new Set(prev);
        newSet.delete(faqId);
        return newSet;
      });
      
      // 如果当前页只有一条数据，删除后应该跳转到上一页
      if (faqs.length === 1 && page > 1) {
        setPage(page - 1);
      } else {
        fetchFAQs();
      }
    } catch (error: any) {
      toast.error(error.message || '删除失败');
    }
  };

  const handleSelectItem = (faqId: string) => {
    setSelectedItems(prev => {
      const newSet = new Set(prev);
      if (newSet.has(faqId)) {
        newSet.delete(faqId);
      } else {
        newSet.add(faqId);
      }
      return newSet;
    });
  };

  const handleSelectAll = () => {
    const currentPageIds = faqs.filter(faq => faq.id).map(faq => faq.id!);
    const allSelected = currentPageIds.every(id => selectedItems.has(id));
    
    if (allSelected) {
      // 取消全选当前页
      setSelectedItems(prev => {
        const newSet = new Set(prev);
        currentPageIds.forEach(id => newSet.delete(id));
        return newSet;
      });
    } else {
      // 全选当前页
      setSelectedItems(prev => {
        const newSet = new Set(prev);
        currentPageIds.forEach(id => newSet.add(id));
        return newSet;
      });
    }
  };

  const handleBatchDelete = async () => {
    if (selectedItems.size === 0) {
      toast.error('请先选择要删除的FAQ');
      return;
    }

    if (!confirm(`确定要删除选中的 ${selectedItems.size} 条FAQ吗？`)) return;

    try {
      const deletePromises = Array.from(selectedItems).map(faqId =>
        tenantFetch(`/api/config/apps/${appId}/faqs/${faqId}`, {
          method: 'DELETE',
        })
      );

      const results = await Promise.all(deletePromises);
      const failedCount = results.filter(res => !res.ok).length;
      const successCount = selectedItems.size - failedCount;

      if (failedCount > 0) {
        toast.error(`删除失败 ${failedCount} 条，成功 ${successCount} 条`);
      } else {
        toast.success(`成功删除 ${successCount} 条FAQ`);
      }

      // 清空选中项
      setSelectedItems(new Set());
      
      // 刷新列表
      fetchFAQs();
    } catch (error: any) {
      toast.error(error.message || '批量删除失败');
    }
  };

  const isAllSelected = faqs.length > 0 && faqs.filter(faq => faq.id).every(faq => selectedItems.has(faq.id!));

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
      const responseData = data.data || [];
      
      // 计算成功上传的文件数（items_count > 0 表示成功提取到FAQ片段）
      const successCount = responseData.filter(
        (item: any) => item.items_count > 0
      ).length;
      
      // 计算总片段数
      const totalChunks = responseData.reduce(
        (sum: number, item: any) => sum + (item.items_count || 0),
        0
      );

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
          enable_question_in_retrieval: faqConfigData?.enable_question_in_retrieval,
          enable_question_in_response: faqConfigData?.enable_question_in_response,
          enable_answer_in_retrieval: faqConfigData?.enable_answer_in_retrieval,
          enable_answer_in_response: faqConfigData?.enable_answer_in_response,
          return_direct: faqConfigData?.return_direct ?? false,
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

  const isFAQActive = faqConfigData?.active ?? false;

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Label htmlFor="enable_faq_switch">开启FAQ回复</Label>
            <Switch
              id="enable_faq_switch"
              checked={isFAQActive}
              onCheckedChange={handleToggleFAQ}
            />
          </div>
          <span className="text-xs text-muted-foreground">
            {isFAQActive ? '如需关闭FAQ功能，请点击关闭FAQ' : '如需使用FAQ功能，请点击开启FAQ'}
          </span>
        </div>
        {isFAQActive && (
          <div className="flex items-center gap-4">
            {faqConfigData && (
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
            {selectedItems.size > 0 && (
              <Button 
                onClick={handleBatchDelete} 
                size="sm" 
                variant="destructive"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                删除选中 ({selectedItems.size})
              </Button>
            )}
          </div>
        )}
      </div>

      {isFAQActive && (
        <>
          {loading ? (
            <div className="text-center py-8 text-muted-foreground">加载中...</div>
          ) : faqs.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              暂无FAQ，点击"新增FAQ"添加
            </div>
          ) : (
            <>
              <div className="mb-4 text-sm text-muted-foreground">
                共 {totalItems} 条FAQ，第 {page} / {totalPages} 页
              </div>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[50px]">
                      <Checkbox
                        checked={isAllSelected}
                        onCheckedChange={handleSelectAll}
                        aria-label="全选"
                      />
                    </TableHead>
                    <TableHead className="w-[200px]">问题</TableHead>
                    <TableHead>答案</TableHead>
                    <TableHead className="w-[120px]">操作</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {faqs.map((faq) => (
                    <TableRow key={faq.id}>
                      <TableCell>
                        <Checkbox
                          checked={faq.id ? selectedItems.has(faq.id) : false}
                          onCheckedChange={() => faq.id && handleSelectItem(faq.id)}
                          aria-label={`选择 ${faq.question}`}
                        />
                      </TableCell>
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
              {totalItems > 0 && (
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-sm text-muted-foreground">
                      显示第 {((page - 1) * pageSize) + 1} - {Math.min(page * pageSize, totalItems)} 条，共 {totalItems} 条
                    </div>
                  </div>
                  <div className="flex justify-center">
                    <PaginationComponent
                      currentPage={page}
                      totalPages={totalPages}
                      onPageChange={handlePageChange}
                    />
                  </div>
                </div>
              )}
            </>
          )}
        </>
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
                      value={[faqConfigData?.score_threshold ?? DEFAULT_SCORE_THRESHOLD]}
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
                      <span className="font-medium">{(faqConfigData?.score_threshold ?? DEFAULT_SCORE_THRESHOLD).toFixed(2)}</span>
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
                      checked={faqConfigData?.enable_question_in_retrieval ?? true}
                      onCheckedChange={(checked) =>
                        setFaqConfigData({ ...faqConfigData!, enable_question_in_retrieval: checked })
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
                      checked={faqConfigData?.enable_question_in_response ?? false}
                      onCheckedChange={(checked) =>
                        setFaqConfigData({ ...faqConfigData!, enable_question_in_response: checked })
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
                      checked={faqConfigData?.enable_answer_in_retrieval ?? false}
                      onCheckedChange={(checked) =>
                        setFaqConfigData({ ...faqConfigData!, enable_answer_in_retrieval: checked })
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
                      checked={faqConfigData?.enable_answer_in_response ?? true}
                      onCheckedChange={(checked) =>
                        setFaqConfigData({ ...faqConfigData!, enable_answer_in_response: checked })
                      }
                    />
                  </div>
                </div>

                {/* 直接返回设置 */}
                <div className="space-y-3">
                  <Label>返回设置</Label>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="return_direct" className="text-sm font-normal">
                        直接返回结果
                      </Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-xs">开启后，FAQ工具将直接返回搜索结果，不经过LLM加工处理</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                    <Switch
                      id="return_direct"
                      checked={faqConfigData?.return_direct ?? false}
                      onCheckedChange={(checked) =>
                        setFaqConfigData({ ...faqConfigData!, return_direct: checked })
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

