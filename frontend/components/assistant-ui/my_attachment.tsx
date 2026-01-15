'use client';

import { PropsWithChildren, useEffect, useState, type FC, useCallback } from 'react';
import { CircleXIcon, FileIcon, PaperclipIcon, PlayCircleIcon, FileTextIcon } from 'lucide-react';
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
import { Loader2, CheckCircle, XCircle } from 'lucide-react';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';

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

// Attachment URL cache and pending requests tracker to avoid repeated API calls
type AttachmentUrlData = { url: string | null; contentType: string | null; fileContent: string | null };
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

    // Create a new request
    const fetchUrl = async (): Promise<AttachmentUrlData> => {
      const defaultData: AttachmentUrlData = { url: null, contentType: null, fileContent: null };
      try {
        const response = await tenantFetch(`/api/config/attachments/urls?ids=${attachmentId}`);
        const result = await response.json();
        
        if (result.code === 200 && result.data?.items?.length > 0) {
          const item = result.data.items[0];
          const newData: AttachmentUrlData = {
            url: item.url || null,
            contentType: item.content_type || null,
            fileContent: item.file_content || null,
          };
          attachmentUrlCache.set(attachmentId, newData);
          return newData;
        }
        return defaultData;
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
      您的浏览器不支持视频播放
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
      return '视频附件预览';
    case 'image':
      return '图片附件预览';
    case 'text':
      return '文本文件预览';
    default:
      return '附件预览';
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
        <DialogDescription>文件名: {fileName}</DialogDescription>
        {loading ? (
          <div className="flex items-center justify-center p-8">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
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

const AttachmentThumb: FC = () => {
  const isImage = useAttachment((a) => a.contentType?.startsWith('image/') ?? false);
  const isVideo = useIsVideo();
  const isFromMessage = useIsFromMessage();
  const attachmentId = useAttachmentId();
  const localContentType = useAttachmentContentType();
  const localImageSrc = useAttachmentSrc();
  
  // Fetch remote data for message attachments
  const { url: remoteUrl, contentType: remoteContentType } = useRemoteAttachmentUrl(attachmentId, isFromMessage);
  
  const effectiveContentType = remoteContentType || localContentType;
  const isText = isTextContentType(effectiveContentType);
  const isEffectiveVideo = isVideo || isVideoContentType(effectiveContentType);
  const isEffectiveImage = isImage || isImageContentType(effectiveContentType);
  
  // Use remote URL for message attachments, local URL for composer attachments
  const thumbSrc = isFromMessage ? remoteUrl : localImageSrc;
  
  return (
    <Avatar className="bg-muted flex size-10 items-center justify-center rounded border text-sm">
      <AvatarFallback delayMs={isEffectiveImage ? 200 : 0}>
        {isEffectiveVideo ? (
          <PlayCircleIcon className="text-primary" />
        ) : isText ? (
          <FileTextIcon className="text-blue-500" />
        ) : (
          <FileIcon />
        )}
      </AvatarFallback>
      {isEffectiveImage && thumbSrc && <AvatarImage src={thumbSrc} />}
    </Avatar>
  );
};

const AttachmentUI: FC = () => {
  const canRemove = useAttachment((a) => a.source !== 'message');
  const uploadStatus = useAttachment((a) => a.status);
  const isFromMessage = useIsFromMessage();
  const typeLabel = useAttachment((a) => {
    const type = a.type;
    switch (type) {
    case 'image':
      return 'Image';
    case 'document':
      return 'Document';
    case 'file':
      // Check if it's a video file
      if (isVideoContentType(a.contentType)) {
        return 'Video';
      }
      return 'File';
    default:
      const _exhaustiveCheck: never = type;
      throw new Error(`Unknown attachment type: ${_exhaustiveCheck}`);
    }
  });
  
  // 安全访问属性
  const progress =
    'progress' in (uploadStatus ?? {})
      ? (uploadStatus as { progress: number }).progress
      : 0;
  const isUploading = uploadStatus.type === 'running' && progress < 100;
  const isError = uploadStatus.type === 'incomplete';
  
  return (
    <Tooltip>
      <AttachmentPrimitive.Root className="relative mt-3">
        <AttachmentPreviewDialog>
          <TooltipTrigger asChild>
            <div className="flex h-12 w-40 items-center justify-center gap-2 rounded-lg border p-1">
              <AttachmentThumb />
              <div className="flex-grow basis-0">
                <p className="text-muted-foreground line-clamp-1 text-ellipsis break-all text-xs font-medium">
                  <AttachmentPrimitive.Name />
                </p>
                <div className="flex felx-row items-center gap-2 py-1">
                  {isError ? (
                    <>
                      {/* 上传失败状态 */}
                      <XCircle className="h-3 w-3 text-red-500" />
                      <span className="text-red-500 text-xs">上传失败</span>
                    </>
                  ) : isUploading ? (
                    <>
                      {/* 上传中状态 */}
                      <Loader2 className="h-3 w-3 animate-spin text-yellow-500" />
                      <span className="text-yellow-500 text-xs">上传中</span>
                    </>
                  ) : (
                    <>
                      {/* 上传完成状态 */}
                      <CheckCircle className="h-3 w-3 text-green-500" />
                      <span className="text-green-500 text-xs">已上传</span>
                    </>
                  ) }
                </div>
              </div>
            </div>
          </TooltipTrigger>
        </AttachmentPreviewDialog>
        {canRemove && <AttachmentRemove />}
      </AttachmentPrimitive.Root>
      <TooltipContent side="top">
        <AttachmentPrimitive.Name />
      </TooltipContent>
    </Tooltip>
  );
};

const AttachmentRemove: FC = () => {
  return (
    <AttachmentPrimitive.Remove asChild>
      <TooltipIconButton
        tooltip=""
        className="absolute -right-3 -top-3 w-6 h-6"
        side="top"
      >
        <CircleXIcon className="text-red-500 hover:text-red-700" />
      </TooltipIconButton>
    </AttachmentPrimitive.Remove>
  );
};

export const UserMessageAttachments: FC = () => {
  return (
    <div className="flex w-full flex-row gap-3 col-span-full col-start-1 row-start-1 justify-end">
      <MessagePrimitive.Attachments components={{ Attachment: AttachmentUI }} />
    </div>
  );
};

export const ComposerAttachments: FC = () => {
  return (
    <div>
      <ComposerPrimitive.Attachments
        components={{ Attachment: AttachmentUI }}
      />
    </div>
  );
};

export const ComposerAddAttachment: FC = () => {
  return (
    <ComposerPrimitive.AddAttachment asChild>
      <TooltipIconButton
        className="my-2.5 w-24 h-8 p-2 transition-opacity ease-in"
        tooltip="上传附件"
        variant="ghost"
      >
        <PaperclipIcon />
        上传附件
      </TooltipIconButton>
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
