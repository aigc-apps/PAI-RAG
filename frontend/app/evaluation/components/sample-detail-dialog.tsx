'use client';

import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogFooter,
    DialogDescription,
} from "@/components/ui/dialog";
import {
    Eye,
    Pencil,
    Trash2Icon,
    Plus,
    Tag,
    MessageSquare,
    CheckCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { useState, useEffect } from "react";
import { SampleItem } from '@/app/evaluation/[datasetId]/types';
import { useI18n } from '@/app/providers/i18n';

interface SampleDetailDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    sample: SampleItem | null;
    mode: 'view' | 'edit';
    onSave?: (updatedSample: SampleItem) => void; // Only needed in edit mode
}

export function SampleDetailDialog({
    open,
    onOpenChange,
    sample,
    mode,
    onSave,
}: SampleDetailDialogProps) {
    const { t } = useI18n();
    const [editedInput, setEditedInput] = useState("");
    const [editedOutput, setEditedOutput] = useState("");
    const [editedMetadata, setEditedMetadata] = useState<Record<string, any>>({});

    // Initialize state when sample changes
    useEffect(() => {
        if (sample) {
            setEditedInput(sample.input || "");
            setEditedOutput(sample.expected_output || "");
            setEditedMetadata(sample.eval_metadata || {});
        }
    }, [sample]);

    if (!sample) return null;

    const handleSave = () => {
        if (!onSave) return;
        const updatedSample: SampleItem = {
            ...sample,
            input: editedInput,
            expected_output: editedOutput,
            eval_metadata: Object.keys(editedMetadata).length > 0 ? editedMetadata : undefined,
        };
        onSave(updatedSample);
    };


    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        {mode === 'view' ? (
                            <>
                                <Eye className="h-5 w-5 text-blue-500" /> {t('evaluation.viewSample')}
                            </>
                        ) : (
                            <>
                                <Pencil className="h-5 w-5 text-green-500" /> {t('evaluation.editSample')}
                            </>
                        )}
                    </DialogTitle>
                    <DialogDescription>
                        {t('evaluation.sampleIdLabel')}: {sample.id}
                    </DialogDescription>
                </DialogHeader>

                <div className="space-y-4 py-4">
                    {/* Question */}
                    <div className="space-y-2">
                        <Label htmlFor="edit-input" className="flex items-center gap-1">
                            <MessageSquare className="h-4 w-4 text-blue-500" />
                            {t('evaluation.questionLabel')}
                        </Label>
                        {mode === 'view' ? (
                            <div className="p-3 bg-muted rounded-md border">
                                <p className="whitespace-pre-wrap">{sample.input}</p>
                            </div>
                        ) : (
                            <Textarea
                                id="edit-input"
                                value={editedInput}
                                onChange={(e) => setEditedInput(e.target.value)}
                                placeholder={t('evaluation.enterQuestion')}
                                className="min-h-[80px]"
                            />
                        )}
                    </div>

                    {/* Answer */}
                    <div className="space-y-2">
                        <Label htmlFor="edit-output" className="flex items-center gap-1">
                            <CheckCircle className="h-4 w-4 text-green-500" />
                            {t('evaluation.answerLabel')}
                        </Label>
                        {mode === 'view' ? (
                            <div className="p-3 bg-green-50 rounded-md border border-green-200">
                                <p className="whitespace-pre-wrap text-green-800">{sample.expected_output}</p>
                            </div>
                        ) : (
                            <Textarea
                                id="edit-output"
                                value={editedOutput}
                                onChange={(e) => setEditedOutput(e.target.value)}
                                placeholder={t('evaluation.enterExpectedAnswer')}
                                className="min-h-[80px]"
                            />
                        )}
                    </div>

                    {/* Dynamic Metadata */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2">
                            <Tag className="h-4 w-4 text-purple-500" />
                            <h3 className="text-sm font-medium">{t('evaluation.metadata')}</h3>
                            {mode === 'edit' && (
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => {
                                        setEditedMetadata(prev => ({
                                            ...prev,
                                            [`${t('evaluation.fieldName')}${Object.keys(prev).length + 1}`]: ""
                                        }));
                                    }}
                                    className="ml-auto"
                                >
                                    <Plus className="h-3 w-3 mr-1" /> {t('evaluation.addField')}
                                </Button>
                            )}
                        </div>

                        {mode === 'view' ? (
                            // View mode
                            sample.eval_metadata && Object.keys(sample.eval_metadata).length > 0 ? (
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                    {Object.entries(sample.eval_metadata).map(([key, value]) => (
                                        <div key={key} className="p-3 bg-purple-50 rounded-md border border-purple-200">
                                            <div className="text-xs font-medium text-purple-600 mb-1">{key}</div>
                                            <div className="text-sm text-purple-800 break-words">
                                                {value !== null && value !== undefined ? String(value) : t('evaluation.emptyValue')}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="p-3 bg-muted rounded-md border text-center text-muted-foreground">
                                    {t('evaluation.noMetadata')}
                                </div>
                            )
                        ) : (
                            // Edit mode
                            editedMetadata && Object.keys(editedMetadata).length > 0 ? (
                                <div className="space-y-3">
                                    {Object.entries(editedMetadata).map(([key, value]) => (
                                        <div key={key} className="flex gap-2 items-start p-2 bg-muted/50 rounded-md">
                                            <Input
                                                value={key}
                                                onChange={(e) => {
                                                    const newMetadata = { ...editedMetadata };
                                                    delete newMetadata[key];
                                                    newMetadata[e.target.value] = value;
                                                    setEditedMetadata(newMetadata);
                                                }}
                                                placeholder={t('evaluation.fieldName')}
                                                className="w-1/3 text-sm"
                                            />
                                            <Input
                                                value={value !== null && value !== undefined ? String(value) : ''}
                                                onChange={(e) => {
                                                    const newMetadata = { ...editedMetadata };
                                                    newMetadata[key] = e.target.value;
                                                    setEditedMetadata(newMetadata);
                                                }}
                                                placeholder={t('evaluation.fieldValue')}
                                                className="flex-1 text-sm"
                                            />
                                            <Button
                                                type="button"
                                                variant="ghost"
                                                size="icon"
                                                className="h-8 w-8 text-red-500 hover:text-red-700 hover:bg-red-100"
                                                onClick={() => {
                                                    const newMetadata = { ...editedMetadata };
                                                    delete newMetadata[key];
                                                    setEditedMetadata(newMetadata);
                                                }}
                                            >
                                                <Trash2Icon className="h-4 w-4" />
                                            </Button>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="p-3 bg-muted/50 rounded-md border-dashed border text-center text-muted-foreground">
                                    {t('evaluation.clickAddFieldToAddMetadata')}
                                </div>
                            )
                        )}
                    </div>
                </div>

                <DialogFooter>
                    <Button
                        variant="outline"
                        onClick={() => onOpenChange(false)}
                    >
                        {t('evaluation.close')}
                    </Button>
                    {mode === 'edit' && (
                        <Button
                            onClick={handleSave}
                            className="bg-green-600 hover:bg-green-700 text-white"
                        >
                            {t('common.save')}
                        </Button>
                    )}
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}