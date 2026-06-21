'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Spinner } from '@/components/ui/loading';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter,
} from '@/components/ui/dialog';
import {
  Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription,
} from '@/components/ui/sheet';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select';
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator,
} from '@/components/ui/dropdown-menu';
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card';
import {
  Plus, RefreshCcw, Clock, FileText, MoreVertical, CheckCircle, XCircle,
  AlertTriangle, Trash2, Pencil, Globe, BookOpen, Power,
} from 'lucide-react';
import { toast } from 'sonner';
import { useI18n } from '@/app/providers/i18n';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';

interface DataSource {
  id: string;
  name: string;
  datasource_key: string;
  source_type: string;
  source_config: Record<string, any>;
  sync_schedule: string | null;
  enabled: boolean;
  status: string;
  last_sync_at: string | null;
  last_sync_finished_at: string | null;
  last_error: string | null;
  doc_count: number;
  last_sync_report: Record<string, any> | null;
  next_sync_at: string | null;
}

interface SyncStatus {
  datasource_id: string;
  status: string;
  doc_count: number;
  total_documents: number;
  synced: number;
  ingesting: number;
  failed: number;
  last_sync_report: Record<string, any> | null;
  last_error: string | null;
}

interface DsDocument {
  doc_id: string;
  title: string | null;
  path: string;
  source_url: string | null;
  byte_size: number | null;
  doc_status: string;
  parse_status: string | null;
  parse_failed_reason: string | null;
  last_changed_at: string | null;
}

const ACTIVE = new Set(['syncing', 'ingesting']);

function formatBytes(n: number | null): string {
  if (!n && n !== 0) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function formatTime(iso: string | null): string {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleString('zh-CN', { hour12: false }).replace(/\//g, '-');
  } catch {
    return iso;
  }
}

export default function DataSourcesPanel({ kbId }: { kbId: string }) {
  const { t } = useI18n();
  const { tenantFetch } = useTenantFetch();

  const [sources, setSources] = useState<DataSource[]>([]);
  const [statusMap, setStatusMap] = useState<Record<string, SyncStatus>>({});
  const [loading, setLoading] = useState(true);

  const [dialogOpen, setDialogOpen] = useState(false);
  const [editing, setEditing] = useState<DataSource | null>(null);

  const [deleteTarget, setDeleteTarget] = useState<DataSource | null>(null);
  const [deleting, setDeleting] = useState(false);

  const [docsTarget, setDocsTarget] = useState<DataSource | null>(null);
  const [docsOpen, setDocsOpen] = useState(false);

  const [cancelTarget, setCancelTarget] = useState<DataSource | null>(null);
  const [cancelling, setCancelling] = useState(false);

  const fetchSources = useCallback(async () => {
    try {
      const res = await tenantFetch(`/api/config/knowledgebases/${kbId}/datasources?page=1&size=100`);
      if (!res.ok) throw new Error(t('datasource.fetchFailed'));
      const json = await res.json();
      setSources(json.data?.items || []);
    } catch (err: any) {
      toast.error(err.message || t('datasource.fetchFailed'));
    } finally {
      setLoading(false);
    }
  }, [kbId, tenantFetch, t]);

  useEffect(() => {
    fetchSources();
  }, [fetchSources]);

  // Poll sync-status only for sources that are actively syncing/ingesting.
  const sourcesRef = useRef(sources);
  sourcesRef.current = sources;
  const statusRef = useRef(statusMap);
  statusRef.current = statusMap;

  useEffect(() => {
    const timer = setInterval(async () => {
      const active = sourcesRef.current.filter((s) =>
        ACTIVE.has(statusRef.current[s.id]?.status ?? s.status),
      );
      if (active.length === 0) return;
      let anyTerminal = false;
      await Promise.all(
        active.map(async (s) => {
          try {
            const res = await tenantFetch(
              `/api/config/knowledgebases/${kbId}/datasources/${s.id}/sync-status`,
            );
            if (!res.ok) return;
            const d = (await res.json()).data as SyncStatus;
            setStatusMap((prev) => ({ ...prev, [s.id]: d }));
            if (!ACTIVE.has(d.status)) anyTerminal = true;
          } catch {
            /* ignore transient poll errors */
          }
        }),
      );
      if (anyTerminal) fetchSources();
    }, 3000);
    return () => clearInterval(timer);
  }, [kbId, tenantFetch, fetchSources]);

  const handleSync = async (s: DataSource) => {
    try {
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/datasources/${s.id}/sync`,
        { method: 'POST' },
      );
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(json?.message || t('datasource.syncFailed'));
      if (json?.data?.status === 'already_syncing') {
        toast.info(t('datasource.alreadySyncing'));
      } else {
        toast.success(t('datasource.syncTriggered'));
      }
      // optimistic: flip to syncing so polling kicks in immediately
      setSources((prev) => prev.map((x) => (x.id === s.id ? { ...x, status: 'syncing' } : x)));
    } catch (err: any) {
      toast.error(err.message || t('datasource.syncFailed'));
    }
  };

  const handleCancel = async () => {
    if (!cancelTarget) return;
    setCancelling(true);
    try {
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/datasources/${cancelTarget.id}/cancel`,
        { method: 'POST' },
      );
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(json?.message || t('datasource.cancelFailed'));
      toast.success(t('datasource.cancelTriggered', { count: json?.data?.cancelled ?? 0 }));
      setCancelTarget(null);
      fetchSources();
    } catch (err: any) {
      toast.error(err.message || t('datasource.cancelFailed'));
    } finally {
      setCancelling(false);
    }
  };

  const handleToggleEnabled = async (s: DataSource) => {
    const action = s.enabled ? 'disable' : 'enable';
    try {
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/datasources/${s.id}/${action}`,
        { method: 'POST' },
      );
      if (!res.ok) throw new Error((await res.json().catch(() => ({})))?.message);
      fetchSources();
    } catch (err: any) {
      toast.error(err.message || t('common.error'));
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/datasources/${deleteTarget.id}`,
        { method: 'DELETE' },
      );
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(json?.message || t('datasource.deleteFailed'));
      toast.success(t('datasource.deleteSuccess'));
      setDeleteTarget(null);
      fetchSources();
    } catch (err: any) {
      toast.error(err.message || t('datasource.deleteFailed'));
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="py-2">
      <div className="flex items-center justify-between px-2 mb-3">
        <span className="text-sm font-medium">{t('datasource.title')}</span>
        <Button
          variant="default"
          className="h-6 text-xs"
          onClick={() => {
            setEditing(null);
            setDialogOpen(true);
          }}
        >
          <Plus className="h-3 w-3" /> {t('datasource.addSource')}
        </Button>
      </div>

      {loading ? (
        <div className="flex justify-center py-12">
          <Spinner size="lg" />
        </div>
      ) : sources.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <Globe className="h-8 w-8 text-muted-foreground mb-3" />
          <p className="text-sm text-muted-foreground mb-1">{t('datasource.emptyTitle')}</p>
          <p className="text-xs text-muted-foreground/70 mb-4">{t('datasource.emptyDesc')}</p>
          <Button
            variant="outline"
            className="h-7 text-xs"
            onClick={() => {
              setEditing(null);
              setDialogOpen(true);
            }}
          >
            <Plus className="h-3 w-3" /> {t('datasource.addSource')}
          </Button>
        </div>
      ) : (
        <div className="flex flex-col gap-2.5 px-2">
          {sources.map((s) => (
            <SourceCard
              key={s.id}
              source={s}
              status={statusMap[s.id]}
              onSync={() => handleSync(s)}
              onEdit={() => {
                // Defer opening so the (closing) dropdown unmounts first —
                // avoids the Radix dropdown→dialog focus/scroll-lock race.
                setEditing(s);
                requestAnimationFrame(() => setDialogOpen(true));
              }}
              onToggle={() => handleToggleEnabled(s)}
              onCancel={() => setCancelTarget(s)}
              onDelete={() => setDeleteTarget(s)}
              onViewDocs={() => { setDocsTarget(s); setDocsOpen(true); }}
            />
          ))}
        </div>
      )}

      <SourceDialog
        kbId={kbId}
        open={dialogOpen}
        editing={editing}
        onOpenChange={setDialogOpen}
        onSaved={() => {
          setDialogOpen(false);
          fetchSources();
        }}
      />

      <DocumentsDrawer
        kbId={kbId}
        open={docsOpen}
        source={docsTarget}
        onOpenChange={setDocsOpen}
      />

      <ConfirmDialog
        open={!!cancelTarget}
        onOpenChange={(o) => !o && setCancelTarget(null)}
        variant="warning"
        title={t('datasource.confirmCancel')}
        description={t('datasource.confirmCancelDesc')}
        target={cancelTarget ? { label: t('datasource.name'), value: cancelTarget.name } : undefined}
        confirmLabel={t('datasource.cancel')}
        cancelLabel={t('datasource.keepRunning')}
        onConfirm={handleCancel}
        loading={cancelling}
      />

      <ConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(o) => !o && setDeleteTarget(null)}
        title={t('datasource.confirmDelete')}
        description={t('datasource.confirmDeleteDesc')}
        target={deleteTarget ? { label: t('datasource.name'), value: deleteTarget.name } : undefined}
        confirmLabel={t('common.delete')}
        onConfirm={handleDelete}
        loading={deleting}
      />
    </div>
  );
}

function StatusBadge({ status, error }: { status: string; error?: string | null }) {
  const { t } = useI18n();
  const base = 'h-5 px-1.5 text-[10px]';
  if (status === 'syncing' || status === 'ingesting') {
    return (
      <Badge variant="secondary" className={`bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900/20 dark:text-blue-400 ${base}`}>
        <Spinner size="sm" className="mr-1" />
        {status === 'syncing' ? t('datasource.statusSyncing') : t('datasource.statusIngesting')}
      </Badge>
    );
  }
  if (status === 'succeeded') {
    return (
      <Badge variant="secondary" className={`bg-green-100 text-green-700 hover:bg-green-200 dark:bg-green-900/20 dark:text-green-400 ${base}`}>
        <CheckCircle className="mr-1 h-3 w-3" />
        {t('datasource.statusSucceeded')}
      </Badge>
    );
  }
  if (status === 'partial') {
    return (
      <Badge variant="secondary" className={`bg-yellow-100 text-yellow-700 hover:bg-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-400 ${base}`}>
        <AlertTriangle className="mr-1 h-3 w-3" />
        {t('datasource.statusPartial')}
      </Badge>
    );
  }
  if (status === 'failed') {
    return (
      <HoverCard>
        <HoverCardTrigger asChild>
          <Badge variant="secondary" className={`bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900/20 dark:text-red-400 cursor-pointer ${base}`}>
            <XCircle className="mr-1 h-3 w-3" />
            {t('datasource.statusFailed')}
          </Badge>
        </HoverCardTrigger>
        {error && <HoverCardContent className="w-80 text-xs">{error}</HoverCardContent>}
      </HoverCard>
    );
  }
  if (status === 'cancelled') {
    return (
      <Badge variant="secondary" className={`bg-muted text-muted-foreground ${base}`}>
        <XCircle className="mr-1 h-3 w-3" />
        {t('datasource.statusCancelled')}
      </Badge>
    );
  }
  return <Badge variant="secondary" className={base}>{t('datasource.statusIdle')}</Badge>;
}

function scheduleLabel(t: (k: string) => string, sched: string | null): string {
  if (!sched) return t('datasource.scheduleManual');
  const map: Record<string, string> = {
    '3600': t('datasource.everyHour'),
    '21600': t('datasource.every6Hours'),
    '86400': t('datasource.everyDay'),
    '604800': t('datasource.everyWeek'),
  };
  return map[sched] || `${sched}s`;
}

function SourceCard({
  source, status, onSync, onEdit, onToggle, onCancel, onDelete, onViewDocs,
}: {
  source: DataSource;
  status?: SyncStatus;
  onSync: () => void;
  onEdit: () => void;
  onToggle: () => void;
  onCancel: () => void;
  onDelete: () => void;
  onViewDocs: () => void;
}) {
  const { t } = useI18n();
  const effStatus = status?.status ?? source.status;
  const isActive = ACTIVE.has(effStatus);
  const Icon = source.source_type === 'sphinx' ? BookOpen : Globe;
  const report = status?.last_sync_report ?? source.last_sync_report;
  const total = status?.total_documents ?? source.doc_count;
  const synced = status?.synced ?? 0;
  const pct = isActive && total > 0 ? Math.min(100, Math.round((synced / total) * 100)) : 0;

  return (
    <div className={`rounded-lg border bg-card p-3 ${effStatus === 'failed' ? 'border-red-200 dark:border-red-900/40' : isActive ? 'border-blue-200 dark:border-blue-900/40' : 'border-border'} ${!source.enabled ? 'opacity-60' : ''}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2.5 min-w-0">
          <Icon className="h-4 w-4 text-muted-foreground shrink-0" />
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-medium truncate">{source.name}</span>
              <Badge variant="secondary" className="h-4 px-1.5 text-[10px] font-mono bg-muted text-muted-foreground">
                {source.source_type}
              </Badge>
              <StatusBadge status={effStatus} error={status?.last_error ?? source.last_error} />
              {!source.enabled && (
                <Badge variant="secondary" className="h-4 px-1.5 text-[10px]">{t('datasource.disabled')}</Badge>
              )}
            </div>
            <div className="text-[11px] text-muted-foreground mt-1 flex items-center gap-1.5 flex-wrap">
              <span>{t('datasource.docCount', { count: total.toLocaleString() })}</span>
              <span>·</span>
              <span>{t('datasource.lastSync')}: {formatTime(source.last_sync_finished_at || source.last_sync_at)}</span>
              <span>·</span>
              <Clock className="h-3 w-3" />
              <span>{scheduleLabel(t, source.sync_schedule)}</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {isActive ? (
            <Button
              variant="outline"
              className="h-6 text-xs text-rose-700 border-rose-200 hover:bg-rose-50 hover:text-rose-800 dark:text-rose-400 dark:border-rose-900/40 dark:hover:bg-rose-900/20"
              onClick={onCancel}
            >
              <XCircle className="h-3 w-3" />
              {t('datasource.cancel')}
            </Button>
          ) : (
            <Button variant="outline" className="h-6 text-xs" onClick={onSync}>
              <RefreshCcw className="h-3 w-3" />
              {t('datasource.syncNow')}
            </Button>
          )}
          <Button variant="outline" className="h-6 text-xs" onClick={onViewDocs}>
            <FileText className="h-3 w-3" />
            {t('datasource.documents')}
          </Button>
          <DropdownMenu modal={false}>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="h-6 w-6 p-0" aria-label={t('datasource.more')}>
                <MoreVertical className="h-3.5 w-3.5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={onEdit}>
                <Pencil className="mr-2 h-3.5 w-3.5" /> {t('common.edit')}
              </DropdownMenuItem>
              <DropdownMenuItem onClick={onToggle}>
                <Power className="mr-2 h-3.5 w-3.5" />
                {source.enabled ? t('datasource.disable') : t('datasource.enable')}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem variant="destructive" onClick={onDelete}>
                <Trash2 className="mr-2 h-3.5 w-3.5" /> {t('common.delete')}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {isActive && (
        <div className="mt-3">
          <div className="h-1.5 rounded-full bg-muted overflow-hidden">
            <div className="h-full bg-blue-500 transition-all duration-500" style={{ width: `${pct}%` }} />
          </div>
          <div className="flex justify-between text-[11px] text-muted-foreground mt-1.5">
            <span>{effStatus === 'syncing' ? t('datasource.phaseFetching') : t('datasource.phaseIngesting')}</span>
            {report?.summary && <span className="font-mono">{report.summary}</span>}
          </div>
        </div>
      )}

      {effStatus === 'failed' && (source.last_error || status?.last_error) && (
        <div className="mt-2 text-[11px] text-red-600 dark:text-red-400 truncate">
          {source.last_error || status?.last_error}
        </div>
      )}
    </div>
  );
}

function SourceDialog({
  kbId, open, editing, onOpenChange, onSaved,
}: {
  kbId: string;
  open: boolean;
  editing: DataSource | null;
  onOpenChange: (open: boolean) => void;
  onSaved: () => void;
}) {
  const { t } = useI18n();
  const { tenantFetch } = useTenantFetch();
  const isEdit = !!editing;

  const [name, setName] = useState('');
  const [key, setKey] = useState('');
  const [type, setType] = useState('llms_txt');
  const [product, setProduct] = useState('');
  const [llmsUrl, setLlmsUrl] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [lang, setLang] = useState('');
  const [schedEnabled, setSchedEnabled] = useState(false);
  const [schedule, setSchedule] = useState('86400');
  const [saving, setSaving] = useState(false);

  // Reset the form from `editing` each time the dialog opens (the component
  // stays mounted across opens, so initializers run only once).
  useEffect(() => {
    if (!open) return;
    setName(editing?.name || '');
    setKey(editing?.datasource_key || '');
    setType(editing?.source_type || 'llms_txt');
    setProduct(editing?.source_config?.product || '');
    setLlmsUrl(editing?.source_config?.llms_url || '');
    setBaseUrl(editing?.source_config?.base_url || '');
    setLang(editing?.source_config?.lang || '');
    setSchedEnabled(!!editing?.sync_schedule);
    setSchedule(editing?.sync_schedule || '86400');
    setSaving(false);
  }, [open, editing]);

  const buildConfig = (): Record<string, any> => {
    if (type === 'llms_txt') {
      const cfg: Record<string, any> = {};
      if (llmsUrl.trim()) cfg.llms_url = llmsUrl.trim();
      else if (product.trim()) cfg.product = product.trim();
      if (lang) cfg.lang = lang;
      return cfg;
    }
    const cfg: Record<string, any> = { base_url: baseUrl.trim() };
    if (lang) cfg.lang = lang;
    return cfg;
  };

  const handleSave = async () => {
    if (!name.trim()) return toast.error(t('datasource.nameRequired'));
    if (!isEdit && !key.trim()) return toast.error(t('datasource.keyRequired'));
    if (type === 'llms_txt' && !product.trim() && !llmsUrl.trim()) {
      return toast.error(t('datasource.llmsConfigRequired'));
    }
    if (type === 'sphinx' && !baseUrl.trim()) {
      return toast.error(t('datasource.baseUrlRequired'));
    }
    setSaving(true);
    try {
      const sync_schedule = schedEnabled ? schedule : null;
      let res: Response;
      if (isEdit) {
        res = await tenantFetch(`/api/config/knowledgebases/${kbId}/datasources/${editing!.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name.trim(), source_config: buildConfig(), sync_schedule }),
        });
      } else {
        res = await tenantFetch(`/api/config/knowledgebases/${kbId}/datasources`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: name.trim(),
            datasource_key: key.trim(),
            source_type: type,
            source_config: buildConfig(),
            sync_schedule,
            enabled: true,
          }),
        });
      }
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(json?.message || t('datasource.saveFailed'));
      toast.success(isEdit ? t('datasource.updateSuccess') : t('datasource.createSuccess'));
      onSaved();
    } catch (err: any) {
      toast.error(err.message || t('datasource.saveFailed'));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{isEdit ? t('datasource.editSource') : t('datasource.addSource')}</DialogTitle>
          <DialogDescription className="text-xs">{t('datasource.dialogDesc')}</DialogDescription>
        </DialogHeader>

        <div className="space-y-3 py-1">
          <div>
            <Label className="text-xs">{t('datasource.name')}</Label>
            <Input className="h-8 text-sm mt-1" value={name} onChange={(e) => setName(e.target.value)} placeholder="EasyRec 文档" />
          </div>

          <div>
            <Label className="text-xs">{t('datasource.key')}</Label>
            <Input
              className="h-8 text-sm mt-1 font-mono"
              value={key}
              disabled={isEdit}
              onChange={(e) => setKey(e.target.value)}
              placeholder="easyrec_docs"
            />
            <p className="text-[11px] text-muted-foreground mt-1">{t('datasource.keyHint')}</p>
          </div>

          <div className="flex gap-2">
            <div className="flex-1">
              <Label className="text-xs">{t('datasource.type')}</Label>
              <Select value={type} onValueChange={setType} disabled={isEdit}>
                <SelectTrigger className="h-8 text-sm mt-1"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="llms_txt">{t('datasource.typeLlms')}</SelectItem>
                  <SelectItem value="sphinx">{t('datasource.typeSphinx')}</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="w-28">
              <Label className="text-xs">{t('datasource.lang')}</Label>
              <Select value={lang || 'auto'} onValueChange={(v) => setLang(v === 'auto' ? '' : v)}>
                <SelectTrigger className="h-8 text-sm mt-1"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">{t('datasource.langAuto')}</SelectItem>
                  <SelectItem value="zh">zh</SelectItem>
                  <SelectItem value="en">en</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {type === 'llms_txt' ? (
            <>
              <div>
                <Label className="text-xs">{t('datasource.product')}</Label>
                <Input className="h-8 text-sm mt-1" value={product} onChange={(e) => setProduct(e.target.value)} placeholder="pai" />
              </div>
              <div>
                <Label className="text-xs">{t('datasource.llmsUrl')}</Label>
                <Input className="h-8 text-sm mt-1" value={llmsUrl} onChange={(e) => setLlmsUrl(e.target.value)} placeholder="https://help.aliyun.com/zh/.../llms.txt" />
                <p className="text-[11px] text-muted-foreground mt-1">{t('datasource.llmsHint')}</p>
              </div>
            </>
          ) : (
            <div>
              <Label className="text-xs">{t('datasource.baseUrl')}</Label>
              <Input className="h-8 text-sm mt-1" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="https://easyrec.readthedocs.io/en/latest/" />
            </div>
          )}

          <div className="flex items-center justify-between rounded-md bg-muted/50 px-3 py-2">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm">{t('datasource.scheduledSync')}</span>
            </div>
            <Switch checked={schedEnabled} onCheckedChange={setSchedEnabled} />
          </div>
          {schedEnabled && (
            <Select value={schedule} onValueChange={setSchedule}>
              <SelectTrigger className="h-8 text-sm"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="3600">{t('datasource.everyHour')}</SelectItem>
                <SelectItem value="21600">{t('datasource.every6Hours')}</SelectItem>
                <SelectItem value="86400">{t('datasource.everyDay')}</SelectItem>
                <SelectItem value="604800">{t('datasource.everyWeek')}</SelectItem>
              </SelectContent>
            </Select>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" className="h-8 text-xs" onClick={() => onOpenChange(false)} disabled={saving}>
            {t('common.cancel')}
          </Button>
          <Button variant="default" className="h-8 text-xs" onClick={handleSave} disabled={saving}>
            {saving && <Spinner size="sm" className="mr-1" />}
            {isEdit ? t('common.save') : t('common.create')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function DocStatusBadge({ doc }: { doc: DsDocument }) {
  const { t } = useI18n();
  const st = doc.parse_status === 'failed' || doc.doc_status === 'failed'
    ? 'failed'
    : doc.doc_status === 'cancelled'
      ? 'cancelled'
      : doc.parse_status === 'succeeded' || doc.doc_status === 'synced'
        ? 'synced'
        : 'ingesting';
  const base = 'h-5 px-1.5 text-[10px]';
  if (st === 'synced') {
    return <Badge variant="secondary" className={`bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400 ${base}`}>{t('datasource.docSynced')}</Badge>;
  }
  if (st === 'cancelled') {
    return <Badge variant="secondary" className={`bg-muted text-muted-foreground ${base}`}>{t('datasource.statusCancelled')}</Badge>;
  }
  if (st === 'failed') {
    return (
      <HoverCard>
        <HoverCardTrigger asChild>
          <Badge variant="secondary" className={`bg-red-100 text-red-700 dark:bg-red-900/20 dark:text-red-400 cursor-pointer ${base}`}>{t('datasource.docFailed')}</Badge>
        </HoverCardTrigger>
        {doc.parse_failed_reason && <HoverCardContent className="w-80 text-xs">{doc.parse_failed_reason}</HoverCardContent>}
      </HoverCard>
    );
  }
  return (
    <Badge variant="secondary" className={`bg-blue-100 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400 ${base}`}>
      <Spinner size="sm" className="mr-1" />{t('datasource.docIngesting')}
    </Badge>
  );
}

function DocumentsDrawer({
  kbId, open, source, onOpenChange,
}: {
  kbId: string;
  open: boolean;
  source: DataSource | null;
  onOpenChange: (open: boolean) => void;
}) {
  const { t } = useI18n();
  const { tenantFetch } = useTenantFetch();
  const [docs, setDocs] = useState<DsDocument[]>([]);
  const [total, setTotal] = useState(0);
  const [pages, setPages] = useState(0);
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState('all');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(true);

  // Reset paging/filters whenever a new source is opened.
  useEffect(() => {
    if (open) {
      setPage(1);
      setStatusFilter('all');
      setQuery('');
    }
  }, [open, source?.id]);

  const fetchDocs = useCallback(async () => {
    if (!open || !source) return;
    setLoading(true);
    try {
      const params = new URLSearchParams({ page: String(page), size: '20' });
      if (statusFilter !== 'all') params.set('doc_status', statusFilter);
      if (query.trim()) params.set('query', query.trim());
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/datasources/${source.id}/documents?${params}`,
      );
      if (!res.ok) throw new Error();
      const json = await res.json();
      setDocs(json.data?.items || []);
      setTotal(json.data?.total || 0);
      setPages(json.data?.pages || 0);
    } catch {
      toast.error(t('datasource.fetchDocsFailed'));
    } finally {
      setLoading(false);
    }
  }, [open, kbId, source?.id, page, statusFilter, query, tenantFetch, t]);

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs]);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="sm:max-w-2xl w-full flex flex-col">
        <SheetHeader>
          <SheetTitle className="text-base">{source?.name} · {t('datasource.documents')}</SheetTitle>
          <SheetDescription className="text-xs">
            {t('datasource.docTotal', { count: total.toLocaleString() })}
          </SheetDescription>
        </SheetHeader>

        <div className="flex items-center gap-2 px-4">
          <Input
            className="h-7 text-xs flex-1"
            placeholder={t('datasource.searchDocs')}
            value={query}
            onChange={(e) => { setQuery(e.target.value); setPage(1); }}
          />
          <Select value={statusFilter} onValueChange={(v) => { setStatusFilter(v); setPage(1); }}>
            <SelectTrigger className="h-7 text-xs w-28"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">{t('datasource.allStatus')}</SelectItem>
              <SelectItem value="synced">{t('datasource.docSynced')}</SelectItem>
              <SelectItem value="ingesting">{t('datasource.docIngesting')}</SelectItem>
              <SelectItem value="failed">{t('datasource.docFailed')}</SelectItem>
              <SelectItem value="cancelled">{t('datasource.statusCancelled')}</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex-1 overflow-y-auto px-4 mt-2">
          {loading ? (
            <div className="flex justify-center py-10"><Spinner size="lg" /></div>
          ) : docs.length === 0 ? (
            <div className="text-center text-xs text-muted-foreground py-10">{t('datasource.noDocs')}</div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">{t('datasource.document')}</TableHead>
                  <TableHead className="text-xs w-20">{t('datasource.status')}</TableHead>
                  <TableHead className="text-xs w-16">{t('datasource.size')}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {docs.map((d) => (
                  <TableRow key={d.doc_id}>
                    <TableCell className="text-xs py-1.5">
                      <div className="truncate max-w-[340px]">
                        {d.source_url ? (
                          <a href={d.source_url} target="_blank" rel="noreferrer" className="hover:underline">
                            {d.title || d.path}
                          </a>
                        ) : (d.title || d.path)}
                      </div>
                      <div className="text-[10px] text-muted-foreground/70 truncate max-w-[340px] font-mono">{d.path}</div>
                    </TableCell>
                    <TableCell className="py-1.5"><DocStatusBadge doc={d} /></TableCell>
                    <TableCell className="text-xs py-1.5 text-muted-foreground">{formatBytes(d.byte_size)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </div>

        {pages > 1 && (
          <div className="flex items-center justify-end gap-2 px-4 py-2 text-xs">
            <Button variant="outline" className="h-6 text-xs" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
              {t('datasource.previous')}
            </Button>
            <span className="text-muted-foreground">{page} / {pages}</span>
            <Button variant="outline" className="h-6 text-xs" disabled={page >= pages} onClick={() => setPage((p) => p + 1)}>
              {t('datasource.next')}
            </Button>
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}
