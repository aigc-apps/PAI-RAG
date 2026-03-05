'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  TrashIcon,
  UploadIcon,
  Sparkles,
  TerminalIcon,
  FileTextIcon,
  PackageIcon,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
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
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { toast } from 'sonner';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

interface Skill {
  id: string;
  name: string;
  description: string;
  skill_type: string;
  enabled: boolean;
  content: string;
  required_tools: string | null;
  required_env: string | null;
  prerequisites: string | null;
}

function SkillTypeIcon({ type }: { type: string }) {
  switch (type) {
    case 'composite':
      return <PackageIcon className="h-4 w-4" />;
    case 'command':
      return <TerminalIcon className="h-4 w-4" />;
    default:
      return <FileTextIcon className="h-4 w-4" />;
  }
}

function SkillTypeBadge({ type, t }: { type: string; t: (key: string) => string }) {
  const variant = type === 'declarative' ? 'secondary' : type === 'composite' ? 'default' : 'outline';
  const label = type === 'declarative' ? t('skills.declarative') : type === 'composite' ? t('skills.composite') : t('skills.command');
  return (
    <Badge variant={variant} className="text-xs">
      <SkillTypeIcon type={type} />
      <span className="ml-1">{label}</span>
    </Badge>
  );
}

export default function SkillsPage() {
  const { t } = useI18n();
  const { tenantFetch } = useTenantFetch();

  const [skills, setSkills] = useState<Skill[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isInstallOpen, setIsInstallOpen] = useState(false);
  const [isInstalling, setIsInstalling] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Skill | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchSkills = useCallback(async () => {
    try {
      setIsLoading(true);
      const res = await tenantFetch('/api/config/skills');
      if (!res.ok) throw new Error('Failed to load skills');
      const data = await res.json();
      setSkills(data.data?.items || []);
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSkills();
  }, [fetchSkills]);

  const handleInstallFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.md')) {
      toast.error('Only .md files are supported.');
      return;
    }

    setIsInstalling(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await tenantFetch('/api/config/skills/install', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.message || t('skills.installFailed'));
      }

      toast.success(t('skills.installSuccess'));
      setIsInstallOpen(false);
      await fetchSkills();
    } catch (err: any) {
      toast.error(err.message || t('skills.installFailed'));
    } finally {
      setIsInstalling(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleInstallFile(file);
    }
    // Reset input so the same file can be selected again
    e.target.value = '';
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleInstallFile(file);
    }
  };

  const handleToggle = async (skill: Skill) => {
    try {
      const res = await tenantFetch(`/api/config/skills/${skill.id}/toggle?enabled=${!skill.enabled}`, {
        method: 'PUT',
      });
      if (!res.ok) throw new Error(t('skills.toggleFailed'));
      toast.success(t('skills.toggleSuccess'));
      setSkills(prev =>
        prev.map(s => s.id === skill.id ? { ...s, enabled: !s.enabled } : s)
      );
    } catch (err: any) {
      toast.error(err.message || t('skills.toggleFailed'));
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      const res = await tenantFetch(`/api/config/skills/${deleteTarget.id}`, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error(t('skills.deleteFailed'));
      toast.success(t('skills.deleteSuccess'));
      setSkills(prev => prev.filter(s => s.id !== deleteTarget.id));
    } catch (err: any) {
      toast.error(err.message || t('skills.deleteFailed'));
    } finally {
      setDeleteTarget(null);
    }
  };

  const parseJsonSafe = (jsonStr: string | null): any[] => {
    if (!jsonStr) return [];
    try {
      return JSON.parse(jsonStr);
    } catch {
      return [];
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Sparkles className="h-6 w-6" />
            <span suppressHydrationWarning>{t('skills.title')}</span>
          </h1>
          <p className="text-sm text-muted-foreground mt-1" suppressHydrationWarning>
            {t('skills.description')}
          </p>
        </div>
        <Button onClick={() => setIsInstallOpen(true)}>
          <UploadIcon className="h-4 w-4 mr-2" />
          <span suppressHydrationWarning>{t('skills.installSkill')}</span>
        </Button>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-20 text-muted-foreground">
          <span suppressHydrationWarning>{t('common.loading')}</span>
        </div>
      ) : skills.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
          <Sparkles className="h-12 w-12 mb-4 opacity-30" />
          <p className="text-lg" suppressHydrationWarning>{t('skills.noSkills')}</p>
          <p className="text-sm" suppressHydrationWarning>{t('skills.noSkillsHint')}</p>
        </div>
      ) : (
        <div className="space-y-3">
          {skills.map(skill => (
            <Card key={skill.id} className={`transition-opacity ${!skill.enabled ? 'opacity-60' : ''}`}>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <CardTitle className="text-base">{skill.name}</CardTitle>
                    <SkillTypeBadge type={skill.skill_type} t={t} />
                  </div>
                  <div className="flex items-center gap-3">
                    <Switch
                      checked={skill.enabled}
                      onCheckedChange={() => handleToggle(skill)}
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-destructive hover:text-destructive"
                      onClick={() => setDeleteTarget(skill)}
                    >
                      <TrashIcon className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                {skill.description && (
                  <CardDescription className="text-sm mt-1">{skill.description}</CardDescription>
                )}
              </CardHeader>
              <CardContent className="pt-0">
                <div className="flex flex-wrap gap-2">
                  {parseJsonSafe(skill.required_tools).map((tool: string) => (
                    <Badge key={tool} variant="outline" className="text-xs font-mono">
                      {tool}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Install Dialog */}
      <Dialog open={isInstallOpen} onOpenChange={setIsInstallOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle suppressHydrationWarning>{t('skills.installDialogTitle')}</DialogTitle>
            <DialogDescription suppressHydrationWarning>
              {t('skills.installDialogDesc')}
            </DialogDescription>
          </DialogHeader>
          <div
            className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:bg-muted/50 transition-colors"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={handleDrop}
          >
            <UploadIcon className="h-8 w-8 mx-auto mb-3 text-muted-foreground" />
            <p className="text-sm text-muted-foreground" suppressHydrationWarning>
              {t('skills.dragHint')}
            </p>
            <p className="text-xs text-muted-foreground mt-1" suppressHydrationWarning>
              {t('skills.uploadHint')}
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".md"
              className="hidden"
              onChange={handleFileSelect}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsInstallOpen(false)} disabled={isInstalling}>
              <span suppressHydrationWarning>{t('common.cancel')}</span>
            </Button>
            <Button onClick={() => fileInputRef.current?.click()} disabled={isInstalling}>
              {isInstalling ? (
                <span suppressHydrationWarning>{t('skills.installing')}</span>
              ) : (
                <span suppressHydrationWarning>{t('skills.uploadFile')}</span>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteTarget} onOpenChange={() => setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle suppressHydrationWarning>{t('common.confirm')}</AlertDialogTitle>
            <AlertDialogDescription suppressHydrationWarning>
              {t('skills.deleteConfirm')}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel suppressHydrationWarning>{t('common.cancel')}</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete} suppressHydrationWarning>
              {t('common.delete')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
