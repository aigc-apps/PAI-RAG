"use client";

import { PropsWithChildren, useEffect, useState, type FC } from "react";
import { CircleXIcon, FileIcon, PaperclipIcon } from "lucide-react";
import {
  AttachmentPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  useAttachment,
} from "@assistant-ui/react";
import { useShallow } from "zustand/shallow";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Dialog,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
  DialogOverlay,
  DialogPortal,
} from "@/components/ui/dialog";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { DialogContent as DialogPrimitiveContent } from "@radix-ui/react-dialog";
import { Loader2, CheckCircle, XCircle } from "lucide-react";
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

const useAttachmentSrc = () => {
  const { file, src } = useAttachment(
    useShallow((a): { file?: File; src?: string } => {
      if (a.type !== "image") return {};
      if (a.file) return { file: a.file };
      const src = a.content?.filter((c) => c.type === "image")[0]?.image;
      if (!src) return {};
      return { src };
    }),
  );

  return useFileSrc(file) ?? src;
};

type AttachmentPreviewProps = {
  src: string;
};

const AttachmentPreview: FC<AttachmentPreviewProps> = ({ src }) => {
  const [isLoaded, setIsLoaded] = useState(false);

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={src}
      style={{
        width: "auto",
        height: "auto",
        maxWidth: "75dvh",
        maxHeight: "75dvh",
        display: isLoaded ? "block" : "none",
        overflow: "clip",
      }}
      onLoad={() => setIsLoaded(true)}
      alt="Preview"
    />
  );
};

const AttachmentPreviewDialog: FC<PropsWithChildren> = ({ children }) => {
  const src = useAttachmentSrc();

  if (!src) return children;

  return (
    <Dialog>
      <DialogTrigger
        className="hover:bg-accent/50 cursor-pointer transition-colors"
        asChild
      >
        {children}
      </DialogTrigger>
      <AttachmentDialogContent>
        <DialogTitle>
          图片附件预览
        </DialogTitle>
        <DialogDescription>
          图片附件预览
        </DialogDescription>
        <AttachmentPreview src={src} />
      </AttachmentDialogContent>
    </Dialog>
  );
};


const AttachmentThumb: FC = () => {
  const isImage = useAttachment((a) => a.type === "image");
  const src = useAttachmentSrc();
  return (
    <Avatar className="bg-muted flex size-10 items-center justify-center rounded border text-sm">
      <AvatarFallback delayMs={isImage ? 200 : 0}>
        <FileIcon />
      </AvatarFallback>
      <AvatarImage src={src} />
    </Avatar>
  );
};

const AttachmentUI: FC = () => {
  const canRemove = useAttachment((a) => a.source !== "message");
  const uploadStatus = useAttachment((a) => a.status);
  const typeLabel = useAttachment((a) => {
    const type = a.type;
    switch (type) {
      case "image":
        return "Image";
      case "document":
        return "Document";
      case "file":
        return "File";
      default:
        const _exhaustiveCheck: never = type;
        throw new Error(`Unknown attachment type: ${_exhaustiveCheck}`);
    }
  });
  console.log("uploadStatus", uploadStatus);
  // 安全访问属性
  const progress =
    "progress" in (uploadStatus ?? {})
      ? (uploadStatus as { progress: number }).progress
      : 0;
  const isUploading = uploadStatus.type === "running" && progress < 100;
  const isError = uploadStatus.type === "incomplete";
  console.log("isUploading", isUploading);
  return (
    <Tooltip>
      <AttachmentPrimitive.Root className="relative mt-3">
        <AttachmentPreviewDialog>
          <TooltipTrigger asChild>
            <div className="flex h-12 w-40 items-center justify-center gap-2 rounded-lg border p-1">
              <AttachmentThumb />
              <div className="flex-grow basis-0">
                <p className="text-muted-foreground line-clamp-1 text-ellipsis break-all text-xs font-bold">
                  <AttachmentPrimitive.Name />
                </p>
                <div className="flex felx-row items-center gap-2 py-1">
                  {/* <p className="text-muted-foreground text-xs">{typeLabel}</p> */}
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
                  )}
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
    <div className="flex w-full flex-row gap-3 overflow-x-auto">
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
