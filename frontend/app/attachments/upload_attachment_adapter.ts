import {
  AttachmentAdapter,
  PendingAttachment,
  CompleteAttachment,
} from "@assistant-ui/react";

export class UploadAttachmentAdapter implements AttachmentAdapter {
  public accept = "*/*";

  public async *add({
    file,
  }: {
    file: File;
  }): AsyncGenerator<PendingAttachment, void> {
    // Validate file size
    const fid = crypto.randomUUID();
    const initialStatus = {
      id: fid,
      type: file.type.startsWith("image/") ? "image" : "document",
      name: file.name,
      contentType: file.type || "application/octet-stream",
      file,
      status: {
        type: "running",
        reason: "uploading",
        progress: 0,
      },
    } as PendingAttachment;
    const errorStatus = {
      id: fid,
      type: file.type.startsWith("image/") ? "image" : "document",
      name: file.name,
      contentType: file.type || "application/octet-stream",
      file,
      status: {
        type: "incomplete",
        reason: "error",
      },
    } as PendingAttachment;

    yield initialStatus;

    const maxSize = 10 * 1024 * 1024; // 10MB limit
    if (file.size > maxSize) {
      yield errorStatus;
      return;
    }

    try {
      // 构造上传请求
      const formData = new FormData();
      formData.append("file_id", fid);
      formData.append("file", file); // 将文件加入 FormData

      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
      const response = await fetch(
        `http://localhost:${port}/v1/config/attachments/upload`,
        {
          method: "POST",
          body: formData, // 自动设置 content-type 为 multipart/form-data
        },
      );

      if (!response.ok) {
        throw new Error("上传失败");
      }

      // 解析响应
      const result = await response.json();
      console.log("result", result);

      // 返回成功状态
      yield {
        id: fid,
        type: file.type.startsWith("image/") ? "image" : "document",
        name: file.name,
        contentType: file.type || "application/octet-stream",
        file,
        status: {
          type: "running",
          reason: "uploading",
          progress: 100,
        },
      };
      return;
    } catch (error) {
      // 返回失败状态
      console.log("error", error);
      yield errorStatus;
      return;
    }
  }
  public async send(
    attachment: PendingAttachment,
  ): Promise<CompleteAttachment> {
    return {
      id: attachment.id,
      type: "document",
      name: attachment.name,
      contentType: attachment.contentType || "application/octet-stream",
      content: [
        {
          type: "text",
          text: attachment.id,
        },
      ],
      status: { type: "complete" },
    };
  }
  public async remove(attachment: PendingAttachment): Promise<void> {
    // Cleanup if needed
  }
}
