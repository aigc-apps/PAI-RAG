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
import { SampleItem } from '@/app/evaluation/[evalId]/types';

interface SampleDetailDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    sample: SampleItem | null;
    mode: 'view' | 'edit';
    onSave?: (updatedSample: SampleItem) => void; // 仅在 edit 模式需要
}

export function SampleDetailDialog({
    open,
    onOpenChange,
    sample,
    mode,
    onSave,
}: SampleDetailDialogProps) {
    const [editedInput, setEditedInput] = useState("");
    const [editedOutput, setEditedOutput] = useState("");
    const [editedMetadata, setEditedMetadata] = useState<Record<string, any>>({});

    // 在组件内，useState 下方添加：
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
                                <Eye className="h-5 w-5 text-blue-500" /> 查看样本
                            </>
                        ) : (
                            <>
                                <Pencil className="h-5 w-5 text-green-500" /> 编辑样本
                            </>
                        )}
                    </DialogTitle>
                    <DialogDescription>
                        样本ID: {sample.id}
                    </DialogDescription>
                </DialogHeader>

                <div className="space-y-4 py-4">
                    {/* 问题 */}
                    <div className="space-y-2">
                        <Label htmlFor="edit-input" className="flex items-center gap-1">
                            <MessageSquare className="h-4 w-4 text-blue-500" />
                            问题
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
                                placeholder="请输入问题"
                                className="min-h-[80px]"
                            />
                        )}
                    </div>

                    {/* 答案 */}
                    <div className="space-y-2">
                        <Label htmlFor="edit-output" className="flex items-center gap-1">
                            <CheckCircle className="h-4 w-4 text-green-500" />
                            答案
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
                                placeholder="请输入预期答案"
                                className="min-h-[80px]"
                            />
                        )}
                    </div>

                    {/* 动态 Metadata */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2">
                            <Tag className="h-4 w-4 text-purple-500" />
                            <h3 className="text-sm font-medium">元数据</h3>
                            {mode === 'edit' && (
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => {
                                        setEditedMetadata(prev => ({
                                            ...prev,
                                            [`新字段${Object.keys(prev).length + 1}`]: ""
                                        }));
                                    }}
                                    className="ml-auto"
                                >
                                    <Plus className="h-3 w-3 mr-1" /> 添加字段
                                </Button>
                            )}
                        </div>

                        {mode === 'view' ? (
                            // 查看模式
                            sample.eval_metadata && Object.keys(sample.eval_metadata).length > 0 ? (
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                    {Object.entries(sample.eval_metadata).map(([key, value]) => (
                                        <div key={key} className="p-3 bg-purple-50 rounded-md border border-purple-200">
                                            <div className="text-xs font-medium text-purple-600 mb-1">{key}</div>
                                            <div className="text-sm text-purple-800 break-words">
                                                {value !== null && value !== undefined ? String(value) : '空值'}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="p-3 bg-muted rounded-md border text-center text-muted-foreground">
                                    无元数据
                                </div>
                            )
                        ) : (
                            // 编辑模式
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
                                                placeholder="字段名"
                                                className="w-1/3 text-sm"
                                            />
                                            <Input
                                                value={value !== null && value !== undefined ? String(value) : ''}
                                                onChange={(e) => {
                                                    const newMetadata = { ...editedMetadata };
                                                    newMetadata[key] = e.target.value;
                                                    setEditedMetadata(newMetadata);
                                                }}
                                                placeholder="字段值"
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
                                    点击“添加字段”按钮添加元数据
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
                        关闭
                    </Button>
                    {mode === 'edit' && (
                        <Button
                            onClick={handleSave}
                            className="bg-green-600 hover:bg-green-700 text-white"
                        >
                            保存
                        </Button>
                    )}
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}