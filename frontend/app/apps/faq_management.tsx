'use client';
import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { useI18n } from '@/app/providers/i18n';
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
import { Plus, Edit, Trash2, Settings, HelpCircle, Upload, MessageSquareQuote, FileQuestion, ChevronDown } from 'lucide-react';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Section } from './chatbot_config';
import { PageLoading } from '@/components/ui/loading';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
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
import { Badge } from '@/components/ui/badge';

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
  onConfigChange: (updates: Partial<Chatbot>) => void;
  onSave: (updatedConfig?: Partial<Chatbot>) => Promise<boolean>;
  saving?: boolean;
}

export const FAQManagement: React.FC<FAQManagementProps> = ({ appId, botConfig, onConfigChange, onSave, saving = false }) => {
  const { t } = useI18n();
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
  const [isListExpanded, setIsListExpanded] = useState(false);
  const [deleteFaqTarget, setDeleteFaqTarget] = useState<FAQItem | null>(null);
  const [showBatchDeleteConfirm, setShowBatchDeleteConfirm] = useState(false);
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
    score_threshold: number;
    embedding_model: string;
    enable_question_in_retrieval: boolean;
    enable_question_in_response: boolean;
    enable_answer_in_retrieval: boolean;
    enable_answer_in_response: boolean;
    return_direct: boolean;
  } | null>(null);
  const { tenantFetch } = useTenantFetch();

  // Sync faqConfigData with botConfig.faq_config
  useEffect(() => {
    if (botConfig.faq_config) {
      setFaqConfigData({
        score_threshold: botConfig.faq_config.similarity_threshold ?? DEFAULT_SCORE_THRESHOLD,
        embedding_model: botConfig.faq_config.embedding_model ?? '',
        enable_question_in_retrieval: botConfig.faq_config.enable_question_in_retrieval ?? true,
        enable_question_in_response: botConfig.faq_config.enable_question_in_response ?? false,
        enable_answer_in_retrieval: botConfig.faq_config.enable_answer_in_retrieval ?? false,
        enable_answer_in_response: botConfig.faq_config.enable_answer_in_response ?? true,
        return_direct: botConfig.faq_config.return_direct ?? false,
      });
    } else {
      setFaqConfigData({
        score_threshold: DEFAULT_SCORE_THRESHOLD,
        embedding_model: '',
        enable_question_in_retrieval: true,
        enable_question_in_response: false,
        enable_answer_in_retrieval: false,
        enable_answer_in_response: true,
        return_direct: false,
      });
    }
  }, [botConfig.faq_config]);

  const handleToggleFAQ = async (checked: boolean) => {
    try {
      onConfigChange({ enable_faq: checked });
      const success = await onSave({ enable_faq: checked });
      if (success) {
        toast.success(checked ? t('apps.faqEnabledToast') : t('apps.faqDisabledToast'));
      }
    } catch (error: any) {
      toast.error(error.message || t('apps.saveFailed'));
    }
  };

  useEffect(() => {
    fetchFAQs();
    fetchEmbeddingModels();
  }, [appId, page]);

  // 当页面切换时，清空选中项
  useEffect(() => {
    setSelectedItems(new Set());
  }, [page]);


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
      toast.error(t('apps.faqFillQuestionAnswer'));
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

      if (!res.ok) throw new Error(t('apps.saveFailed'));
      
      toast.success(editingFaq ? t('messages.saveSuccess') : t('apps.createSuccess'));
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
      toast.error(error.message || t('apps.saveFailed'));
    }
  };

  const handleDelete = (faq: FAQItem) => {
    // Open the unified confirm dialog instead of native confirm()
    setDeleteFaqTarget(faq);
  };

  const performDelete = async (faqId: string) => {
    try {
      const res = await tenantFetch(`/api/config/apps/${appId}/faqs/${faqId}`, {
        method: 'DELETE',
      });

      if (!res.ok) throw new Error(t('apps.deleteError'));

      toast.success(t('messages.deleteSuccess'));

      // 从选中项中移除
      setSelectedItems((prev) => {
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
      toast.error(error.message || t('apps.deleteError'));
    } finally {
      setDeleteFaqTarget(null);
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

  const handleBatchDelete = () => {
    if (selectedItems.size === 0) {
      toast.error(t('apps.faqSelectToDelete'));
      return;
    }
    setShowBatchDeleteConfirm(true);
  };

  const performBatchDelete = async () => {
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
        toast.error(t('apps.faqBatchDeletePartial', { failed: String(failedCount), success: String(successCount) }));
      } else {
        toast.success(t('apps.faqBatchDeleteSuccess', { count: String(successCount) }));
      }

      // 清空选中项
      setSelectedItems(new Set());

      // 刷新列表
      fetchFAQs();
    } catch (error: any) {
      toast.error(error.message || t('apps.faqBatchDeleteFailed'));
    } finally {
      setShowBatchDeleteConfirm(false);
    }
  };

  const isAllSelected = faqs.length > 0 && faqs.filter(faq => faq.id).every(faq => selectedItems.has(faq.id!));

  const handleUploadFiles = async () => {
    if (uploadFiles.length === 0) {
      toast.error(t('apps.faqSelectFileToUpload'));
      return;
    }

    // 验证文件类型
    const validFiles = uploadFiles.filter(
      (file) =>
        file.name.endsWith('.xlsx') || file.name.endsWith('.xls')
    );

    if (validFiles.length === 0) {
      toast.error(t('apps.faqSelectValidExcel'));
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
        throw new Error(errorData.message || t('apps.faqUploadFailed'));
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
        t('apps.faqUploadSuccess', { success: String(successCount), total: String(validFiles.length), chunks: String(totalChunks) })
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
      toast.error(error.message || t('apps.faqUploadFailed'));
    } finally {
      setUploading(false);
    }
  };

  const handleSaveConfig = async () => {
    if (!faqConfigData) return;
    
    try {
      const updatedFaqConfig = {
        similarity_threshold: faqConfigData.score_threshold,
        embedding_model: faqConfigData.embedding_model,
        enable_question_in_retrieval: faqConfigData.enable_question_in_retrieval,
        enable_question_in_response: faqConfigData.enable_question_in_response,
        enable_answer_in_retrieval: faqConfigData.enable_answer_in_retrieval,
        enable_answer_in_response: faqConfigData.enable_answer_in_response,
        return_direct: faqConfigData.return_direct,
      };
      
      onConfigChange({ faq_config: updatedFaqConfig });
      const success = await onSave({ faq_config: updatedFaqConfig });
      
      if (success) {
        setIsConfigDialogOpen(false);
        toast.success(t('apps.faqConfigSaveSuccess'));
      }
    } catch (error: any) {
      toast.error(error.message || t('apps.faqConfigSaveFailed'));
    }
  };

  const isFAQActive = botConfig.enable_faq ?? false;

  return (
    <div className="max-w-5xl mx-auto px-6">
      <Section
        icon={<MessageSquareQuote className="w-4 h-4" />}
        title={t('apps.faqEnableReply')}
        description={isFAQActive ? undefined : t('apps.faqDisabledDesc')}
        rightSlot={
          <div className="flex items-center gap-3">
            {isFAQActive && (
              <Badge variant="secondary" className="font-normal">
                {t('apps.faqNumTip', { faqNum: String(totalItems) })}
              </Badge>
            )}
            <Switch
              id="enable_faq_switch"
              checked={isFAQActive}
              onCheckedChange={handleToggleFAQ}
            />
          </div>
        }
      >
        {isFAQActive && (
          <Collapsible open={isListExpanded} onOpenChange={setIsListExpanded}>
            <div className="flex items-center justify-between gap-2 flex-wrap">
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="sm" className="gap-2 -ml-2">
                  <ChevronDown
                    className={`w-4 h-4 transition-transform ${isListExpanded ? '' : '-rotate-90'}`}
                  />
                  {isListExpanded ? t('apps.faqCollapseList') : t('apps.faqExpandList')}
                </Button>
              </CollapsibleTrigger>
              <div className="flex items-center gap-2 flex-wrap">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setIsConfigDialogOpen(true)}
                >
                  <Settings className="w-4 h-4 mr-2" />
                  {t('common.settings')}
                </Button>
                <Button onClick={() => { setIsListExpanded(true); handleOpenDialog(); }} size="sm">
                  <Plus className="w-4 h-4 mr-2" />
                  {t('apps.faqAddNew')}
                </Button>
                <Button onClick={() => setIsUploadDialogOpen(true)} size="sm" variant="outline">
                  <Upload className="w-4 h-4 mr-2" />
                  {t('apps.faqUpload')}
                </Button>
                {selectedItems.size > 0 && (
                  <Button
                    onClick={handleBatchDelete}
                    size="sm"
                    variant="destructive"
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    {t('apps.faqDeleteSelected', { count: String(selectedItems.size) })}
                  </Button>
                )}
              </div>
            </div>
            <CollapsibleContent className="mt-4">
          {loading ? (
            <PageLoading />
          ) : faqs.length === 0 ? (
            <div className="empty-state">
              <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-primary/10 text-primary mb-4">
                <FileQuestion className="w-7 h-7" />
              </div>
              <p className="text-base font-semibold mb-1">{t('apps.faqEmptyTitle')}</p>
              <p className="text-sm text-muted-foreground text-center max-w-md">
                {t('apps.faqEmptyDesc')}
              </p>
            </div>
          ) : (
            <>
              <div className="rounded-lg border border-border overflow-hidden bg-background">
                <div className="px-4 py-2.5 border-b border-border bg-muted/30 text-sm text-muted-foreground">
                  {t('apps.faqPaginationInfo', { total: String(totalItems), page: String(page), pages: String(totalPages) })}
                </div>
                <Table className="table-modern">
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[50px] pl-4">
                        <Checkbox
                          checked={isAllSelected}
                          onCheckedChange={handleSelectAll}
                          aria-label={t('apps.faqSelectAll')}
                        />
                      </TableHead>
                      <TableHead className="w-[260px]">{t('apps.faqQuestion')}</TableHead>
                      <TableHead>{t('apps.faqAnswer')}</TableHead>
                      <TableHead className="w-[120px] text-right pr-4">{t('apps.faqOperation')}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {faqs.map((faq) => (
                      <TableRow key={faq.id}>
                        <TableCell className="pl-4">
                          <Checkbox
                            checked={faq.id ? selectedItems.has(faq.id) : false}
                            onCheckedChange={() => faq.id && handleSelectItem(faq.id)}
                            aria-label={`${t('apps.faqSelectAll')} ${faq.question}`}
                          />
                        </TableCell>
                        <TableCell className="font-medium">{faq.question}</TableCell>
                        <TableCell className="max-w-md truncate text-muted-foreground">{faq.answer}</TableCell>
                        <TableCell className="pr-4">
                          <div className="flex gap-1 justify-end">
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8"
                              onClick={() => handleOpenDialog(faq)}
                            >
                              <Edit className="w-4 h-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 hover:text-destructive"
                              onClick={() => faq.id && handleDelete(faq)}
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
              {totalItems > 0 && (
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-sm text-muted-foreground">
                      {t('apps.faqShowingRange', { start: String(((page - 1) * pageSize) + 1), end: String(Math.min(page * pageSize, totalItems)), total: String(totalItems) })}
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
            </CollapsibleContent>
          </Collapsible>
        )}
      </Section>

      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle>{editingFaq ? t('apps.faqEdit') : t('apps.faqAddNew')}</DialogTitle>
            <DialogDescription>
              {t('apps.faqFormDesc')}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="question">{t('apps.faqQuestion')} *</Label>
              <Input
                id="question"
                value={formData.question}
                onChange={(e) =>
                  setFormData({ ...formData, question: e.target.value })
                }
                placeholder={t('apps.faqQuestionPlaceholder')}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="answer">{t('apps.faqAnswer')} *</Label>
              <Textarea
                id="answer"
                value={formData.answer}
                onChange={(e) =>
                  setFormData({ ...formData, answer: e.target.value })
                }
                placeholder={t('apps.faqAnswerPlaceholder')}
                rows={6}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={handleCloseDialog}>
              {t('common.cancel')}
            </Button>
            <Button onClick={handleSave}>
              {editingFaq ? t('common.update') : t('common.create')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* FAQ配置对话框 */}
      <Dialog open={isConfigDialogOpen} onOpenChange={setIsConfigDialogOpen}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle>{t('apps.faqReplySettings')}</DialogTitle>
            <DialogDescription>
              {t('apps.faqConfigDesc')}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-6 py-4">
            {!faqConfigData ? (
              <div className="text-center py-4 text-muted-foreground">{t('apps.faqLoadingConfig')}</div>
            ) : (
              <>
                {/* 分数阈值 */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="score_threshold">{t('apps.faqScoreThreshold')}</Label>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="text-xs">{t('apps.faqScoreThresholdHint')}</p>
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
                      <span>{t('apps.faqScoreEasyMatch')}</span>
                      <span className="font-medium">{(faqConfigData?.score_threshold ?? DEFAULT_SCORE_THRESHOLD).toFixed(2)}</span>
                      <span>{t('apps.faqScorePreciseMatch')}</span>
                    </div>
                  </div>
                </div>

                {/* Embedding模型 */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="embedding_model">{t('apps.faqEmbeddingModel')}</Label>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="text-xs">{t('apps.faqEmbeddingModelHint')}</p>
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
                      <SelectValue placeholder={t('apps.faqSelectEmbeddingModel')} />
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
                  <Label>{t('apps.faqQuestionParticipation')}</Label>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="question_in_retrieval" className="text-sm font-normal">
                        {t('apps.faqQuestionInRetrieval')}
                      </Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-xs">{t('apps.faqQuestionInRetrievalHint')}</p>
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
                        {t('apps.faqQuestionInResponse')}
                      </Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-xs">{t('apps.faqQuestionInResponseHint')}</p>
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
                  <Label>{t('apps.faqAnswerParticipation')}</Label>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="answer_in_retrieval" className="text-sm font-normal">
                        {t('apps.faqAnswerInRetrieval')}
                      </Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-xs">{t('apps.faqAnswerInRetrievalHint')}</p>
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
                        {t('apps.faqAnswerInResponse')}
                      </Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-xs">{t('apps.faqAnswerInResponseHint')}</p>
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
                  <Label>{t('apps.faqReturnSettings')}</Label>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="return_direct" className="text-sm font-normal">
                        {t('apps.faqReturnDirect')}
                      </Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-xs">{t('apps.faqReturnDirectHint')}</p>
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
              {t('common.cancel')}
            </Button>
            <Button onClick={handleSaveConfig}>
              {t('common.save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* 上传文件对话框 */}
      <Dialog open={isUploadDialogOpen} onOpenChange={setIsUploadDialogOpen}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle>{t('apps.faqUpload')}</DialogTitle>
            <DialogDescription>
              {t('apps.faqSelectFileToUploadDesc')}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            {/* 文件选择 */}
            <div className="space-y-2">
              <Label htmlFor="file-upload">{t('apps.faqSelectFile')}</Label>
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
                  <span className="text-sm font-medium">{t('apps.faqClickToSelectFile')}</span>
                  <span className="text-xs text-muted-foreground">{t('apps.faqSupportedFileTypes')}</span>
                </label>
                {uploadFiles.length > 0 && (
                  <div className="mt-4 space-y-2 text-left">
                    <div className="text-xs text-muted-foreground mb-2">{t('apps.faqFilesSelectedCount', { count: String(uploadFiles.length) })}</div>
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
              <Label>{t('apps.faqFileParseConfig')}</Label>
              
              {/* 标题行下标 */}
              <div className="space-y-2">
                <Label htmlFor="header_index_max" className="text-sm">
                  {t('apps.faqHeaderRowIndex')}
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
                  placeholder={t('apps.faqHeaderRowPlaceholder')}
                />
                <p className="text-xs text-muted-foreground">
                  {t('apps.faqHeaderRowHint')}
                </p>
              </div>

              {/* 问题列 */}
              <div className="space-y-2">
                <Label htmlFor="question_column_index" className="text-sm">
                  {t('apps.faqQuestionColumn')}
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
                  placeholder={t('apps.faqDefaultZero')}
                />
              </div>

              {/* 答案列 */}
              <div className="space-y-2">
                <Label htmlFor="answer_column_index" className="text-sm">
                  {t('apps.faqAnswerColumn')}
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
                  placeholder={t('apps.faqDefaultOne')}
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
              {t('common.cancel')}
            </Button>
            <Button
              onClick={handleUploadFiles}
              disabled={uploadFiles.length === 0 || uploading}
            >
              {uploading ? t('apps.faqUploading') : t('common.upload')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Single FAQ delete confirmation */}
      <ConfirmDialog
        open={!!deleteFaqTarget}
        onOpenChange={(o) => !o && setDeleteFaqTarget(null)}
        title={t('apps.faqConfirmDeleteOne')}
        target={deleteFaqTarget ? { value: deleteFaqTarget.question } : undefined}
        onConfirm={() => {
          if (deleteFaqTarget?.id) performDelete(deleteFaqTarget.id);
        }}
      />

      {/* Batch FAQ delete confirmation */}
      <ConfirmDialog
        open={showBatchDeleteConfirm}
        onOpenChange={setShowBatchDeleteConfirm}
        title={t('apps.faqConfirmDeleteSelected', { count: String(selectedItems.size) })}
        onConfirm={performBatchDelete}
      />
    </div>
  );
};

