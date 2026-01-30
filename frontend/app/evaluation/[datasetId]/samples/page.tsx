'use client';

import React, { useState, useEffect, use } from "react";
import { useRouter } from "next/navigation";
import { useI18n } from '@/app/providers/i18n';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from '@/components/ui/button';
import { PaginationComponent } from "@/components/customized/pagination/pagination-component";
import {
    ChevronDown,
    ChevronUp,
    PlayIcon,
    CopyIcon,
    UploadIcon,
    Eye,
    Trash2Icon,
    Loader2,
    Pencil,
    FileText,
    Info,
    BookAIcon,
} from "lucide-react";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
    DialogDescription,
    DialogFooter,
} from "@/components/ui/dialog";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Badge } from '@/components/ui/badge';
import { RunConfig } from '@/app/evaluation/[datasetId]/types';
import { EvaluatorConfig } from '@/app/evaluation/[datasetId]/types';

import { toast } from 'sonner';

import { SampleDetailDialog } from '@/app/evaluation/components/sample-detail-dialog';
import { useDatasetActions } from '@/app/evaluation/[datasetId]/samples/useDatasetActions';
import { SampleItem } from '@/app/evaluation/[datasetId]/types';
import { useTenantFetch } from "@/hooks/use-tenant-fetch";

export default function EvalDatasetsDetailsPage({
    params,
}: {
    params: Promise<{ datasetId: string }>;
}) {
  
    const { datasetId } = use(params);
    const router = useRouter();
    const { t } = useI18n();

    // === State management ===
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [datasets, setDatasets] = useState<SampleItem[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const pageSize = 10;

    const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
    const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
    const [allItems, setAllItems] = useState<SampleItem[]>([]);
    const [totalItems, setTotalItems] = useState(0);

    // Dialog state
    const [isRunSingleDetailOpen, setIsRunSingleDetailOpen] = useState(false);
    const [isRunBatchDetailOpen, setIsRunBatchDetailOpen] = useState(false);
    const [isEditOpen, setIsEditOpen] = useState(false);
    const [editingSample, setEditingSample] = useState<SampleItem | null>(null);
    const [dialogMode, setDialogMode] = useState<'view' | 'edit'>('view');

    // Experiment configuration related
    const [experimentName, setExperimentName] = useState("");
    const [experimentDescription, setExperimentDescription] = useState("");
    const [runConfigId, setRunConfigId] = useState<string>("");
    const [runConfigs, setRunConfigs] = useState<RunConfig[]>([]);
    const [evaluatorConfigId, setEvaluatorConfigId] = useState<string>("");
    const [evaluatorConfigs, setEvaluatorConfigs] = useState<EvaluatorConfig[]>([]);


    // Upload state
    const [uploading, setUploading] = useState(false);
    const fileInputRef = React.useRef<HTMLInputElement>(null);
    const { tenantFetch } = useTenantFetch();
    const { runSamples, deleteSample, uploadFile } = useDatasetActions({ datasetId });

    // === Data loading ===
    useEffect(() => {
        const loadData = async () => {
            setIsLoading(true);
            try {
                const [datasetRes, runConfigsRes, evalConfigsRes] = await Promise.all([
                    tenantFetch(`/api/config/evaluation/${datasetId}/samples?page=${page}&size=${pageSize}`),
                    tenantFetch(`/api/config/evaluation/${datasetId}/runconfigs`),
                    tenantFetch(`/api/config/evaluation/${datasetId}/evalconfigs`),
                ]);

                // Load paginated data
                if (datasetRes.ok) {
                    const data = await datasetRes.json();
                    setDatasets(data.data.items);
                    setTotalItems(data.data.total);
                    setTotalPages(data.data.pages);
                }

                // Load configurations
                if (runConfigsRes.ok) {
                    const configData = await runConfigsRes.json();
                    setRunConfigs(configData.data.items);
                }

                if (evalConfigsRes.ok) {
                    const evalConfigData = await evalConfigsRes.json();
                    setEvaluatorConfigs(evalConfigData.data.items);
                }

                // Load all data (for select all)
                const tmpPageSize = 1000;
                const firstPageRes = await tenantFetch(`/api/config/evaluation/${datasetId}/samples?page=1&size=${tmpPageSize}`);
                if (!firstPageRes.ok) throw new Error(t('evaluation.fetchSampleListFailed'));
                const json_data = await firstPageRes.json();
                const tmpAllItems: SampleItem[] = [];

                for (let curPage = 1; curPage <= json_data.data.pages; curPage++) {
                    console.log(t('evaluation.loadAllDataPage', { page: curPage }));
                    const response = await tenantFetch(`/api/config/evaluation/${datasetId}/samples?page=${curPage}&size=${tmpPageSize}`);
                    const data = await response.json();
                    tmpAllItems.push(...data.data.items);
                }
                setAllItems(tmpAllItems);


            } catch (err) {
                console.error(err);
                toast.error(t('evaluation.loadDataFailed'));
            } finally {
                setIsLoading(false);
            }
        };

        loadData();
    }, [page, datasetId, datasets.length, t, tenantFetch]);

    // === Interaction functions ===
    const handlePageChange = (newPage: number) => {
        if (newPage < 1 || newPage > totalPages) return;
        setPage(newPage);
    };

    const toggleRowExpansion = (id: string) => {
        setExpandedRows(prev => {
            const newSet = new Set(prev);
            newSet.has(id) ? newSet.delete(id) : newSet.add(id);
            return newSet;
        });
    };

    const copyId = (id: string) => {
        navigator.clipboard.writeText(id);
        toast.success(t('evaluation.copySuccess'));
    };

    const isItemSelected = (id: string) => selectedItems.has(id);

    const handleSelectItem = (id: string) => {
        setSelectedItems(prev => {
            const newSet = new Set(prev);
            newSet.has(id) ? newSet.delete(id) : newSet.add(id);
            return newSet;
        });
    };

    const isAllSelected = selectedItems.size === allItems.length && allItems.length > 0;

    // === Run logic ===
    const handleBatchRun = async () => {
        const ids = isAllSelected
            ? allItems.map(item => item.id)
            : Array.from(selectedItems);

        await runSamples({
            name: experimentName,
            description: experimentDescription,
            sample_ids: ids,
            run_config_id: runConfigId,
            evaluator_config_id: evaluatorConfigId
        });

        setIsRunBatchDetailOpen(false);
        setExperimentName("");
        setExperimentDescription("");
    };

    const runSingleSample = async (id: string) => {
        await runSamples({
            name: experimentName,
            description: experimentDescription,
            sample_ids: [id],
            run_config_id: runConfigId,
            evaluator_config_id: evaluatorConfigId
        });
        setIsRunSingleDetailOpen(false);
        setExperimentName("");
        setExperimentDescription("");
    };

    // === Upload logic ===
    const handleFileUpload = async (files: FileList | null) => {
        if (!files?.length) return;

        const file = files[0];
        if (file.size > 1000 * 1024 * 1024) {
            toast.error(t('evaluation.fileSizeExceeded'));
            return;
        }

        setUploading(true);
        try {
            const uploadedItems = await uploadFile(file);
            console.log('##handleFileUpload', uploadedItems);
            setDatasets(prev => [...prev, ...uploadedItems]);
        } finally {
            setUploading(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    // === Edit/view logic ===
    const handleEyeClick = (item: SampleItem) => {
        setEditingSample(item);
        setDialogMode('view');
        setIsEditOpen(true);
    };

    const handleEditClick = (item: SampleItem) => {
        setEditingSample(item);
        setDialogMode('edit');
        setIsEditOpen(true);
    };

    const handleSaveEdit = async (updatedSample: SampleItem) => {
        try {
            const response = await tenantFetch(`/api/config/evaluation/${datasetId}/samples/${updatedSample.id}`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(updatedSample),
            });

            if (!response.ok) throw new Error(t('evaluation.updateFailed'));

            setDatasets(prev => prev.map(item => item.id === updatedSample.id ? updatedSample : item));
            toast.success(t('evaluation.updateSuccess'));
            setIsEditOpen(false);
        } catch (error) {
            toast.error(t('evaluation.updateFailed'));
        }
    };

    // === Render helper functions ===
    const modifyRunConfig = (selected_ids: Set<string>) => (
        <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="name" className="text-right">{t('evaluation.experimentNameLabel')}</Label>
                <Input
                    id="name"
                    value={experimentName}
                    onChange={(e) => setExperimentName(e.target.value)}
                    className="col-span-3"
                    placeholder={t('evaluation.experimentNamePlaceholder')}
                />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="description" className="text-right">{t('evaluation.descriptionLabel')}</Label>
                <Textarea
                    id="description"
                    value={experimentDescription}
                    onChange={(e) => setExperimentDescription(e.target.value)}
                    className="col-span-3"
                    placeholder={t('evaluation.descriptionPlaceholder')}
                    rows={3}
                />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
                <Label className="text-right">{t('evaluation.dataSampleIds')}</Label>
                <div className="col-span-2">
                    <div className="max-h-40 overflow-y-auto rounded-md border p-2 bg-muted/20">
                        <div className="flex flex-wrap gap-1.5">
                            {[...selected_ids].map(id => (
                                <Badge key={id} variant="secondary" className="bg-green-50 text-green-700">
                                    {id.substring(0, 8)}...
                                </Badge>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
                <Label className="text-right">{t('evaluation.runSettingsLabel')}</Label>
                <div className="grid gap-4 py-1">
                    {runConfigs.length > 0 ? (
                        <Select onValueChange={setRunConfigId}>
                            <SelectTrigger>
                                <SelectValue placeholder={t('evaluation.selectRunSettings')} />
                            </SelectTrigger>
                            <SelectContent>
                                {runConfigs.map(config => (
                                    <SelectItem key={config.id} value={config.id}>
                                        {config.name}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    ) : (
                        <p className="text-sm text-muted-foreground">{t('evaluation.noRunConfig')}</p>
                    )}
                </div>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
                <Label className="text-right">{t('evaluation.evaluatorSettingsLabel')}</Label>
                <div className="grid gap-4 py-1">
                    {evaluatorConfigs.length > 0 ? (
                        <Select onValueChange={setEvaluatorConfigId}>
                            <SelectTrigger>
                                <SelectValue placeholder={t('evaluation.selectEvaluatorSettings')} />
                            </SelectTrigger>
                            <SelectContent>
                                {evaluatorConfigs.map(config => (
                                    <SelectItem key={config.id} value={config.id}>
                                        {config.name}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    ) : (
                        <p className="text-sm text-muted-foreground">{t('evaluation.noEvaluatorConfig')}</p>
                    )}
                </div>
            </div>
        </div>
    );

    // // === UI 渲染 ===
    // if (isLoading) {
    //     return (
    //         <div className="flex items-center justify-center h-64">
    //             <Loader2 className="h-8 w-8 animate-spin" />
    //         </div>
    //     );
    // }

    return (
        <div className="flex flex-col h-full min-h-0">
            {/* Reusable dialog */}
            <SampleDetailDialog
                open={isEditOpen}
                onOpenChange={setIsEditOpen}
                sample={editingSample}
                mode={dialogMode}
                onSave={dialogMode === 'edit' ? handleSaveEdit : undefined}
            />

            <Card className="flex flex-col h-full min-h-0 shadow-sm hover:shadow-md transition-shadow overflow-hidden">
                <CardHeader className="shrink-0 flex md:items-center md:justify-between">
                    <div>
                        <CardTitle className="text-lg font-medium flex items-center gap-2">
                            <FileText className="h-5 w-5" /> {t('evaluation.sampleManagement')}
                        </CardTitle>
                        <p className="text-sm text-muted-foreground mt-1">
                            {t('evaluation.sampleManagementDesc')}
                        </p>
                    </div>
                    <div className="flex flex-col sm:flex-row gap-3 w-full md:w-auto">
                        <div className="flex flex-col gap-2">
                            <div className="flex gap-2 justify-end">
                                {/* 批量运行 */}
                                <Dialog open={isRunBatchDetailOpen} onOpenChange={setIsRunBatchDetailOpen}>
                                    <DialogTrigger asChild>
                                        <Button disabled={selectedItems.size === 0 && !isAllSelected} className="text-white">
                                            <PlayIcon className="mr-2 h-4 w-4" />
                                            {isAllSelected ? t('evaluation.runExperimentAll', { count: totalItems }) : t('evaluation.runExperimentSelected', { count: selectedItems.size })}
                                        </Button>
                                    </DialogTrigger>
                                    <DialogContent className="sm:max-w-[750px]">
                                        <DialogHeader>
                                            <DialogTitle>{t('evaluation.createNewExperimentBatch')}</DialogTitle>
                                            <DialogDescription>{t('evaluation.experimentNameDesc')}</DialogDescription>
                                        </DialogHeader>
                                        {modifyRunConfig(selectedItems)}
                                        <DialogFooter>
                                            <Button variant="outline" onClick={() => setIsRunBatchDetailOpen(false)}>
                                                {t('common.cancel')}
                                            </Button>
                                            <Button onClick={handleBatchRun} disabled={!experimentName.trim()} className="text-white">
                                                {t('evaluation.run')}
                                            </Button>
                                        </DialogFooter>
                                    </DialogContent>
                                </Dialog>

                                {/* Upload */}
                                <Button
                                    onClick={() => document.getElementById('file-upload')?.click()}
                                    disabled={uploading}
                                    className="text-white"
                                >
                                    {uploading ? (
                                        <>
                                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                            {t('evaluation.uploading')}
                                        </>
                                    ) : (
                                        <>
                                            <UploadIcon className="mr-2 h-4 w-4" /> {t('evaluation.importData')}
                                        </>
                                    )}
                                </Button>
                                <input
                                    id="file-upload"
                                    type="file"
                                    className="hidden"
                                    ref={fileInputRef}
                                    onChange={(e) => handleFileUpload(e.target.files)}
                                />
                            </div>

                            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                                <div className="flex items-start gap-2">
                                    <Info className="h-4 w-4 text-gray-600 mt-0.5" />
                                    <p className="text-xs text-gray-800">
                                        <span className="font-medium">{t('evaluation.dataRequirements')}</span>
                                        {t('evaluation.dataRequirementsDesc')}
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </CardHeader>

                <CardContent className="flex-1 min-h-0 overflow-y-auto p-0">
                    <div className="rounded-md h-full min-h-0">
                        <Table className='rounded-md border'>
                            <TableHeader>
                                <TableRow>
                                    <TableHead className="w-[5%]">
                                        <Checkbox
                                            checked={isAllSelected}
                                            onCheckedChange={(checked) => {
                                                setSelectedItems(checked ? new Set(allItems.map(item => item.id)) : new Set());
                                            }}
                                        />
                                    </TableHead>
                                    <TableHead className="w-[10%]">{t('evaluation.sampleId')}</TableHead>
                                    <TableHead className="w-[35%]">{t('evaluation.question')}</TableHead>
                                    <TableHead className="w-[20%]">{t('evaluation.answer')}</TableHead>
                                    <TableHead className="w-[15%]">{t('evaluation.attachment')}</TableHead>
                                    <TableHead className="w-[15%] text-center">{t('evaluation.operations')}</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {isLoading ? (
                                    <TableRow>
                                        <TableCell colSpan={9} className="h-32 text-center">
                                            <div className="flex items-center justify-center space-x-4">
                                                <Loader2 className="h-6 w-6 animate-spin" />
                                                <h4 className="font-medium">Loading Datasets</h4>
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                ) : datasets.length === 0 ? (
                                    <TableRow>
                                        <TableCell colSpan={9} className="h-32 text-center">
                                            <div className="flex flex-col items-center gap-2 text-muted-foreground">
                                                <BookAIcon className="h-8 w-8" />
                                                <span>{t('evaluation.noDataSamples')}</span>
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                ) : (
                                    datasets.map((item) => (
                                        <TableRow key={item.id}>
                                            <TableCell>
                                                <Checkbox
                                                    checked={isItemSelected(item.id)}
                                                    onCheckedChange={() => handleSelectItem(item.id)}
                                                />
                                            </TableCell>
                                            <TableCell className="font-medium">
                                                <div className="flex">
                                                    <Button
                                                        variant="link"
                                                        className="truncate max-w-[120px] font-medium"
                                                        onClick={() => handleEyeClick(item)}
                                                    >
                                                        {item.id.substring(0, 12)}...
                                                    </Button>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        className="h-6 w-6"
                                                        onClick={() => copyId(item.id)}
                                                        title={t('evaluation.copySampleId')}
                                                    >
                                                        <CopyIcon className="h-3 w-3" />
                                                    </Button>
                                                </div>
                                            </TableCell>
                                            <TableCell
                                                className="whitespace-normal break-words min-w-[250px] max-w-[400px] px-4 py-2"
                                                style={{
                                                    whiteSpace: expandedRows.has(item.id) ? 'normal' : 'nowrap',
                                                    overflow: 'hidden',
                                                    textOverflow: 'ellipsis',
                                                }}
                                            >
                                                {item.input}
                                            </TableCell>
                                            <TableCell className="whitespace-normal break-words max-w-[150px] px-4"
                                                style={{
                                                    whiteSpace: expandedRows.has(item.id) ? 'normal' : 'nowrap',
                                                    overflow: 'hidden',
                                                    textOverflow: 'ellipsis',
                                                }}
                                                >
                                                {item.expected_output}
                                            </TableCell>
                                            <TableCell>
                                                <Badge variant="secondary">
                                                    {item.eval_metadata?.file_name}
                                                </Badge>
                                            </TableCell>
                                            <TableCell className="text-right">
                                                <div className="flex items-center justify-end gap-1">
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        onClick={() => toggleRowExpansion(item.id)}
                                                        title={expandedRows.has(item.id) ? t('evaluation.collapse') : t('evaluation.expand')}
                                                    >
                                                        {expandedRows.has(item.id) ? (
                                                            <ChevronUp className="h-4 w-4" />
                                                        ) : (
                                                            <ChevronDown className="h-4 w-4" />
                                                        )}
                                                    </Button>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        onClick={() => handleEyeClick(item)}
                                                        title={t('evaluation.viewDetails')}
                                                    >
                                                        <Eye className="h-4 w-4" />
                                                    </Button>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        className="text-blue-500"
                                                        onClick={() => handleEditClick(item)}
                                                        title={t('evaluation.edit')}
                                                    >
                                                        <Pencil className="h-4 w-4" />
                                                    </Button>
                                                    {/* Single run */}
                                                    <Dialog open={isRunSingleDetailOpen} onOpenChange={setIsRunSingleDetailOpen}>
                                                        <DialogTrigger asChild>
                                                            <Button variant="ghost" size="icon" title={t('evaluation.runSingle')}>
                                                                <PlayIcon className="h-4 w-4" />
                                                            </Button>
                                                        </DialogTrigger>
                                                        <DialogContent className="sm:max-w-[750px]">
                                                            <DialogHeader>
                                                                <DialogTitle>{t('evaluation.createNewExperimentSingle')}</DialogTitle>
                                                                <DialogDescription>{t('evaluation.experimentNameDesc')}</DialogDescription>
                                                            </DialogHeader>
                                                            {modifyRunConfig(new Set([item.id]))}
                                                            <DialogFooter>
                                                                <Button variant="outline" onClick={() => setIsRunSingleDetailOpen(false)}>
                                                                    {t('common.cancel')}
                                                                </Button>
                                                                <Button onClick={() => runSingleSample(item.id)} disabled={!experimentName.trim()} className="text-white">
                                                                    {t('evaluation.run')}
                                                                </Button>
                                                            </DialogFooter>
                                                        </DialogContent>
                                                    </Dialog>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        className="text-red-500"
                                                        onClick={() => deleteSample(item.id)}
                                                        title={t('common.delete')}
                                                    >
                                                        <Trash2Icon className="h-4 w-4" />
                                                    </Button>
                                                </div>
                                            </TableCell>
                                        </TableRow>
                                    ))
                                )}
                            </TableBody>
                        </Table>
                    </div>

                    {(selectedItems.size > 0 || isAllSelected) && (
                        <div className="p-3 border-t">
                            <div className="text-sm flex items-center justify-between">
                                <span className="font-medium">
                                    {isAllSelected ? t('evaluation.selectedAll', { count: totalItems }) : t('evaluation.selectedCount', { count: selectedItems.size })}
                                </span>
                                <Button variant="link" className="p-0 h-auto" onClick={() => setSelectedItems(new Set())}>
                                    {t('evaluation.clearSelection')}
                                </Button>
                            </div>
                        </div>
                    )}
                </CardContent>

                <CardFooter className="shrink-0 border-t pb-2">
                    <PaginationComponent
                        currentPage={page}
                        totalPages={totalPages}
                        onPageChange={handlePageChange}
                    />
                </CardFooter>
            </Card>
        </div>
    );
}