'use client';

import { PropsWithChildren, useEffect, useState, type FC, useCallback } from 'react';
import {
  CircleXIcon, FileIcon, PaperclipIcon, PlayCircleIcon, FileTextIcon,
  FileCodeIcon, FileSpreadsheetIcon, ImageIcon, PresentationIcon,
} from 'lucide-react';
import {
  AttachmentPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  useAttachment,
} from '@assistant-ui/react';
import { useShallow } from 'zustand/shallow';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Dialog,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
  DialogOverlay,
  DialogPortal,
} from '@/components/ui/dialog';
import { Avatar, AvatarImage, AvatarFallback } from '@/components/ui/avatar';
import { TooltipIconButton } from '@/components/assistant-ui/tooltip-icon-button';
import { DialogContent as DialogPrimitiveContent } from '@radix-ui/react-dialog';
import { CheckCircle, XCircle } from 'lucide-react';
import { Spinner } from '@/components/ui/loading';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

// Check if content type is video
const isVideoContentType = (contentType?: string): boolean => {
  return contentType?.startsWith('video/') ?? false;
};

// Check if content type is image
const isImageContentType = (contentType?: string): boolean => {
  return contentType?.startsWith('image/') ?? false;
};

// Check if content type is text-based
const isTextContentType = (contentType?: string): boolean => {
  if (!contentType) return false;
  return contentType.startsWith('text/') ||
         contentType === 'application/json' ||
         contentType === 'application/xml';
};

// Map mime / filename to a branded icon + accent color. Keeps the generic
// grey `FileIcon` as a last resort for unknown types.
type FileVisual = {
  Icon: typeof FileIcon;
  // Tailwind classes; `-on-bubble` palette goes over the user's purple bubble.
  tint: string;
  tintOnBubble: string;
};

const FILE_VISUAL_BY_EXT: Record<string, FileVisual> = {
  pdf:  { Icon: FileTextIcon,        tint: 'bg-rose-500/15 text-rose-600',    tintOnBubble: 'bg-white/20 text-rose-100' },
  doc:  { Icon: FileTextIcon,        tint: 'bg-sky-500/15 text-sky-600',      tintOnBubble: 'bg-white/20 text-sky-100' },
  docx: { Icon: FileTextIcon,        tint: 'bg-sky-500/15 text-sky-600',      tintOnBubble: 'bg-white/20 text-sky-100' },
  ppt:  { Icon: PresentationIcon,    tint: 'bg-orange-500/15 text-orange-600', tintOnBubble: 'bg-white/20 text-orange-100' },
  pptx: { Icon: PresentationIcon,    tint: 'bg-orange-500/15 text-orange-600', tintOnBubble: 'bg-white/20 text-orange-100' },
  xls:  { Icon: FileSpreadsheetIcon, tint: 'bg-emerald-500/15 text-emerald-600', tintOnBubble: 'bg-white/20 text-emerald-100' },
  xlsx: { Icon: FileSpreadsheetIcon, tint: 'bg-emerald-500/15 text-emerald-600', tintOnBubble: 'bg-white/20 text-emerald-100' },
  csv:  { Icon: FileSpreadsheetIcon, tint: 'bg-emerald-500/15 text-emerald-600', tintOnBubble: 'bg-white/20 text-emerald-100' },
  md:   { Icon: FileTextIcon,        tint: 'bg-slate-500/15 text-slate-600',  tintOnBubble: 'bg-white/20 text-slate-100' },
  txt:  { Icon: FileTextIcon,        tint: 'bg-slate-500/15 text-slate-600',  tintOnBubble: 'bg-white/20 text-slate-100' },
  json: { Icon: FileCodeIcon,        tint: 'bg-violet-500/15 text-violet-600', tintOnBubble: 'bg-white/20 text-violet-100' },
  xml:  { Icon: FileCodeIcon,        tint: 'bg-violet-500/15 text-violet-600', tintOnBubble: 'bg-white/20 text-violet-100' },
  yaml: { Icon: FileCodeIcon,        tint: 'bg-violet-500/15 text-violet-600', tintOnBubble: 'bg-white/20 text-violet-100' },
  yml:  { Icon: FileCodeIcon,        tint: 'bg-violet-500/15 text-violet-600', tintOnBubble: 'bg-white/20 text-violet-100' },
  py:   { Icon: FileCodeIcon,        tint: 'bg-violet-500/15 text-violet-600', tintOnBubble: 'bg-white/20 text-violet-100' },
  js:   { Icon: FileCodeIcon,        tint: 'bg-violet-500/15 text-violet-600', tintOnBubble: 'bg-white/20 text-violet-100' },
  ts:   { Icon: FileCodeIcon,        tint: 'bg-violet-500/15 text-violet-600', tintOnBubble: 'bg-white/20 text-violet-100' },
};

const DEFAULT_VISUAL: FileVisual = {
  Icon: FileIcon,
  tint: 'bg-primary/10 text-primary',
  tintOnBubble: 'bg-white/20 text-white',
};

function visualFor(name?: string | null, contentType?: string | null): FileVisual {
  if (contentType?.startsWith('image/')) {
    return { Icon: ImageIcon, tint: 'bg-indigo-500/15 text-indigo-600', tintOnBubble: 'bg-white/20 text-indigo-100' };
  }
  if (contentType?.startsWith('video/')) {
    return { Icon: PlayCircleIcon, tint: 'bg-fuchsia-500/15 text-fuchsia-600', tintOnBubble: 'bg-white/20 text-fuchsia-100' };
  }
  const ext = name?.toLowerCase().split('.').pop() ?? '';
  return FILE_VISUAL_BY_EXT[ext] ?? DEFAULT_VISUAL;
}

function formatBytes(bytes: number | null | undefined): string | null {
  if (bytes == null || bytes < 0) return null;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

// Attachment URL cache and pending requests tracker to avoid repeated API calls
type AttachmentUrlData = {
  url: string | null;
  contentType: string | null;
  fileContent: string | null;
  fileSize: number | null;
};
const attachmentUrlCache = new Map<string, AttachmentUrlData>();
const pendingRequests = new Map<string, Promise<AttachmentUrlData>>();

const useFileSrc = (file: File | undefined) => {
  const [src, setSrc] = useState<string | undefined>(undefined);

  useEffect(() => {
    if (!file) {
      setSrc(undefined);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setSrc(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  return src;
};

// Hook to fetch attachment URL from API (with deduplication)
const useRemoteAttachmentUrl = (attachmentId: string | undefined, isFromMessage: boolean) => {
  const [data, setData] = useState<AttachmentUrlData>({
    url: null,
    contentType: null,
    fileContent: null,
    fileSize: null,
  });
  const [loading, setLoading] = useState(false);
  const { tenantFetch } = useTenantFetch();

  useEffect(() => {
    if (!attachmentId || !isFromMessage) {
      return;
    }

    // Check cache first
    const cached = attachmentUrlCache.get(attachmentId);
    if (cached) {
      setData(cached);
      return;
    }

    // Check if there's already a pending request for this attachment
    const existingRequest = pendingRequests.get(attachmentId);
    if (existingRequest) {
      // Wait for the existing request to complete
      setLoading(true);
      existingRequest.then((result) => {
        setData(result);
        setLoading(false);
      }).catch(() => {
        setLoading(false);
      });
      return;
    }

    // Create a new request. The /v1/files API splits what the old
    // /v1/config/attachments/urls endpoint bundled into three calls; we do
    // metadata + signed URL in parallel, then pull a text preview only when
    // the mime type warrants one.
    const fetchUrl = async (): Promise<AttachmentUrlData> => {
      const defaultData: AttachmentUrlData = {
        url: null, contentType: null, fileContent: null, fileSize: null,
      };
      try {
        const [metaResp, urlResp] = await Promise.all([
          tenantFetch(`/api/files/${attachmentId}`),
          tenantFetch(`/api/files/${attachmentId}/url`),
        ]);
        const metaJson = await metaResp.json();
        const urlJson = await urlResp.json();

        const contentType: string | null = metaJson?.data?.mime_type ?? null;
        const fileSize: number | null =
          typeof metaJson?.data?.file_size === 'number'
            ? metaJson.data.file_size
            : null;
        const url: string | null = urlJson?.data?.url ?? null;

        let fileContent: string | null = null;
        if (isTextContentType(contentType ?? undefined)) {
          try {
            // Cap the inline preview at 5KB — clients paginate via
            // `?offset=&limit=` if they need the rest.
            const textResp = await tenantFetch(
              `/api/files/${attachmentId}/text?limit=5000`,
            );
            const textJson = await textResp.json();
            if (textJson?.code === 200) {
              fileContent = textJson.data?.content ?? null;
            }
          } catch (err) {
            // Text not ready yet (extraction still running, or format has no
            // text form). UI falls back to "no preview" — not an error.
            console.warn('[attachment] text preview unavailable:', err);
          }
        }

        const newData: AttachmentUrlData = {
          url,
          contentType,
          fileContent,
          fileSize,
        };
        attachmentUrlCache.set(attachmentId, newData);
        return newData;
      } catch (error) {
        console.error('Failed to fetch attachment URL:', error);
        return defaultData;
      }
    };

    setLoading(true);
    const requestPromise = fetchUrl();
    pendingRequests.set(attachmentId, requestPromise);
    
    requestPromise.then((result) => {
      setData(result);
      setLoading(false);
      pendingRequests.delete(attachmentId);
    }).catch(() => {
      setLoading(false);
      pendingRequests.delete(attachmentId);
    });
  }, [attachmentId, isFromMessage, tenantFetch]);

  return { ...data, loading };
};

const useAttachmentSrc = () => {
  const { file, src } = useAttachment(
    useShallow((a): { file?: File; src?: string } => {
      if (a.type !== 'image') return {};
      if (a.file) return { file: a.file };
      const src = a.content?.filter((c) => c.type === 'image')[0]?.image;
      if (!src) return {};
      return { src };
    }),
  );

  return useFileSrc(file) ?? src;
};

// Hook for video source from local file
const useVideoSrc = () => {
  const { file } = useAttachment(
    useShallow((a): { file?: File; contentType?: string } => {
      if (a.type === 'file' && isVideoContentType(a.contentType)) {
        return { file: a.file, contentType: a.contentType };
      }
      return {};
    }),
  );

  return useFileSrc(file);
};

// Check if current attachment is a video
const useIsVideo = () => {
  return useAttachment((a) => {
    return isVideoContentType(a.contentType)});
};

// Check if current attachment is from a message (not from composer)
const useIsFromMessage = () => {
  return useAttachment((a) => a.source === 'message');
};

// Get attachment ID
const useAttachmentId = () => {
  return useAttachment((a) => a.id);
};

// Get attachment name
const useAttachmentName = () => {
  return useAttachment((a) => a.name);
};

// Get attachment content type
const useAttachmentContentType = () => {
  return useAttachment((a) => a.contentType);
};

type AttachmentPreviewProps = {
  src: string;
};

const ImagePreview: FC<AttachmentPreviewProps> = ({ src }) => {
  const [isLoaded, setIsLoaded] = useState(false);

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={src}
      style={{
        width: 'auto',
        height: 'auto',
        maxWidth: '75dvh',
        maxHeight: '75dvh',
        display: isLoaded ? 'block' : 'none',
        overflow: 'clip',
      }}
      onLoad={() => setIsLoaded(true)}
      alt="Preview"
    />
  );
};

const VideoPreview: FC<AttachmentPreviewProps> = ({ src }) => {
  const { t } = useI18n();
  return (
    <video
      src={src}
      controls
      autoPlay={false}
      style={{
        width: 'auto',
        height: 'auto',
        maxWidth: '75dvh',
        maxHeight: '75dvh',
        overflow: 'clip',
      }}
    >
      {t('chat.attachment.videoNotSupported')}
    </video>
  );
};

type TextPreviewProps = {
  content: string;
};

const TextPreview: FC<TextPreviewProps> = ({ content }) => {
  return (
    <div
      className="overflow-auto bg-gray-50 rounded p-4 font-mono text-sm"
      style={{
        maxWidth: '75dvh',
        maxHeight: '75dvh',
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
      }}
    >
      {content}
    </div>
  );
};

const AttachmentPreviewDialog: FC<PropsWithChildren> = ({ children }) => {
  const { t } = useI18n();
  const localImageSrc = useAttachmentSrc();
  const localVideoSrc = useVideoSrc();
  const isVideo = useIsVideo();
  const isFromMessage = useIsFromMessage();
  const attachmentId = useAttachmentId();
  const fileName = useAttachmentName();
  const localContentType = useAttachmentContentType();
  
  // Fetch remote URL for message attachments
  const { url: remoteUrl, contentType: remoteContentType, fileContent, loading } = useRemoteAttachmentUrl(
    attachmentId,
    isFromMessage
  );

  // Determine the effective content type and preview type
  const effectiveContentType = remoteContentType || localContentType;
  const isRemoteVideo = isVideoContentType(effectiveContentType);
  const isRemoteImage = isImageContentType(effectiveContentType);
  const isRemoteText = isTextContentType(effectiveContentType);

  // Determine the source to use
  let src: string | null = null;
  let previewType: 'image' | 'video' | 'text' | null = null;

  if (isFromMessage) {
    // For message attachments, use remote URL
    if (isRemoteText && fileContent) {
      previewType = 'text';
    } else if (remoteUrl) {
      src = remoteUrl;
      previewType = isRemoteVideo ? 'video' : isRemoteImage ? 'image' : null;
    }
  } else {
    // For composer attachments, use local file
    if (isVideo && localVideoSrc) {
      src = localVideoSrc;
      previewType = 'video';
    } else if (localImageSrc) {
      src = localImageSrc;
      previewType = 'image';
    }
  }

  // No preview available
  if (!previewType || (previewType !== 'text' && !src)) {
    return children;
  }


  const getTitle = () => {
    switch (previewType) {
    case 'video':
      return t('chat.attachment.preview');
    case 'image':
      return t('chat.attachment.preview');
    case 'text':
      return t('chat.attachment.preview');
    default:
      return t('chat.attachment.preview');
    }
  };

  return (
    <Dialog>
      <DialogTrigger
        className="hover:bg-accent/50 cursor-pointer transition-colors"
        asChild
      >
        {children}
      </DialogTrigger>
      <AttachmentDialogContent>
        <DialogTitle>{getTitle()}</DialogTitle>
        <DialogDescription>{t('chat.attachment.fileName')}: {fileName}</DialogDescription>
        {loading ? (
          <div className="flex items-center justify-center p-8">
            <Spinner size="lg" />
          </div>
        ) : previewType === 'text' && fileContent ? (
          <TextPreview content={fileContent} />
        ) : previewType === 'video' && src ? (
          <VideoPreview src={src} />
        ) : previewType === 'image' && src ? (
          <ImagePreview src={src} />
        ) : null}
      </AttachmentDialogContent>
    </Dialog>
  );
};

const AttachmentThumb: FC<{ inBubble: boolean }> = ({ inBubble }) => {
  const isFromMessage = useIsFromMessage();
  const attachmentId = useAttachmentId();
  const attachmentName = useAttachmentName();
  const localContentType = useAttachmentContentType();
  const localImageSrc = useAttachmentSrc();

  const { url: remoteUrl, contentType: remoteContentType } = useRemoteAttachmentUrl(
    attachmentId,
    isFromMessage,
  );

  const effectiveContentType = remoteContentType || localContentType;
  const isEffectiveImage = (effectiveContentType ?? '').startsWith('image/');

  const thumbSrc = isFromMessage ? remoteUrl : localImageSrc;
  const visual = visualFor(attachmentName, effectiveContentType);
  const tint = inBubble ? visual.tintOnBubble : visual.tint;
  const { Icon } = visual;

  return (
    <Avatar
      className={`flex size-8 items-center justify-center rounded-md overflow-hidden shrink-0 ${tint} [&_svg]:w-4 [&_svg]:h-4`}
    >
      <AvatarFallback delayMs={isEffectiveImage ? 200 : 0} className="bg-transparent">
        <Icon />
      </AvatarFallback>
      {isEffectiveImage && thumbSrc && <AvatarImage src={thumbSrc} />}
    </Avatar>
  );
};

const AttachmentUI: FC = () => {
  const { t } = useI18n();
  const canRemove = useAttachment((a) => a.source !== 'message');
  const uploadStatus = useAttachment((a) => a.status);
  const isFromMessage = useIsFromMessage();

  // Local File (from composer) already has the size; message attachments
  // fetch it alongside mime/url via the cached hook.
  const localFileSize = useAttachment((a) => a.file?.size ?? null);
  const attachmentId = useAttachmentId();
  const { fileSize: remoteFileSize } = useRemoteAttachmentUrl(
    attachmentId, isFromMessage,
  );
  const displaySize = formatBytes(
    isFromMessage ? remoteFileSize : localFileSize,
  );

  const progress =
    'progress' in (uploadStatus ?? {})
      ? (uploadStatus as { progress: number }).progress
      : 0;
  const isUploading = uploadStatus.type === 'running' && progress < 100;
  const isError = uploadStatus.type === 'incomplete';

  // Palette adapts to context: inside the blue user bubble we use
  // translucent white; in the light-colored composer we use muted tones.
  const cardClass = isFromMessage
    ? 'bg-white/12 border-white/25 hover:bg-white/18 hover:shadow-sm'
    : 'bg-muted/40 border-border hover:bg-muted/60 hover:shadow-sm';
  const nameClass = isFromMessage ? 'text-white/95' : 'text-foreground/90';
  const metaClass = isFromMessage ? 'text-white/70' : 'text-muted-foreground';
  const statusColor = isError
    ? 'text-rose-300'
    : isUploading
      ? 'text-amber-200'
      : isFromMessage
        ? 'text-emerald-200'
        : 'text-emerald-600';

  return (
    <Tooltip>
      <AttachmentPrimitive.Root className="relative">
        <AttachmentPreviewDialog>
          <TooltipTrigger asChild>
            <div
              className={`flex items-center gap-2 px-2 py-1.5 rounded-lg border transition-all cursor-pointer w-[220px] ${cardClass}`}
            >
              <AttachmentThumb inBubble={isFromMessage} />
              <div className="flex-1 min-w-0">
                <p
                  className={`line-clamp-1 text-ellipsis break-all text-[11px] font-medium ${nameClass}`}
                >
                  <AttachmentPrimitive.Name />
                </p>
                <div className={`flex items-center gap-1.5 mt-0.5 text-[10px] ${statusColor}`}>
                  {isError ? (
                    <>
                      <XCircle className="h-2.5 w-2.5" />
                      <span>{t('chat.attachment.uploadFailed')}</span>
                    </>
                  ) : isUploading ? (
                    <>
                      <Spinner size="sm" />
                      <span>{t('chat.attachment.uploading')}</span>
                    </>
                  ) : (
                    <>
                      <CheckCircle className="h-2.5 w-2.5" />
                      <span>{t('chat.attachment.uploaded')}</span>
                    </>
                  )}
                  {displaySize && !isError && !isUploading && (
                    <>
                      <span className={`${metaClass}`}>·</span>
                      <span className={`${metaClass}`}>{displaySize}</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          </TooltipTrigger>
        </AttachmentPreviewDialog>
        {canRemove && <AttachmentRemove />}
      </AttachmentPrimitive.Root>
      <TooltipContent side="top" className="text-xs">
        <AttachmentPrimitive.Name />
      </TooltipContent>
    </Tooltip>
  );
};

const AttachmentRemove: FC = () => {
  return (
    <AttachmentPrimitive.Remove asChild>
      <button
        type="button"
        className="absolute -right-1.5 -top-1.5 w-4 h-4 rounded-full bg-background border border-border flex items-center justify-center text-muted-foreground hover:text-destructive hover:border-destructive transition-colors shadow-sm"
        title="Remove"
      >
        <CircleXIcon className="w-3 h-3" />
      </button>
    </AttachmentPrimitive.Remove>
  );
};

export const UserMessageAttachments: FC = () => {
  return (
    <div className="flex flex-wrap gap-1.5 justify-end empty:hidden [&:not(:empty)]:mb-1">
      <MessagePrimitive.Attachments components={{ Attachment: AttachmentUI }} />
    </div>
  );
};

export const ComposerAttachments: FC = () => {
  return (
    <div className="flex flex-wrap gap-1.5">
      <ComposerPrimitive.Attachments
        components={{ Attachment: AttachmentUI }}
      />
    </div>
  );
};

export const ComposerAddAttachment: FC = () => {
  const { t } = useI18n();
  return (
    <ComposerPrimitive.AddAttachment asChild>
      <button
        type="button"
        title={t('chat.attachment.uploadAttachment')}
        className="inline-flex items-center gap-1 h-6 px-1.5 rounded-md border border-input bg-background shadow-xs text-[11px] text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
      >
        <PaperclipIcon className="w-3 h-3" />
        <span>{t('chat.attachment.uploadAttachment')}</span>
      </button>
    </ComposerPrimitive.AddAttachment>
  );
};

const AttachmentDialogContent: FC<PropsWithChildren> = ({ children }) => (
  <DialogPortal>
    <DialogOverlay />
    <DialogPrimitiveContent className="data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%] fixed left-[50%] top-[50%] z-50 grid translate-x-[-50%] translate-y-[-50%] shadow-lg duration-200 bg-white rounded-lg p-6">
      {children}
    </DialogPrimitiveContent>
  </DialogPortal>
);
