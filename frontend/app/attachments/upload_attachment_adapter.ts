import {
  AttachmentAdapter,
  PendingAttachment,
  CompleteAttachment,
} from '@assistant-ui/react';
import { toast } from 'sonner';

export class UploadAttachmentAdapter implements AttachmentAdapter {
  public accept = '*/*';

  public async *add({
    file,
  }: {
    file: File;
  }): AsyncGenerator<PendingAttachment, void> {
    // Validate file size
    const { v4: uuidv4 } = require('uuid');
    const fid = uuidv4();
    yield {
      id: fid,
      type: file.type.startsWith('image/') ? 'image' : 'document',
      name: file.name,
      contentType: file.type || 'application/octet-stream',
      file,
      status: {
        type: 'running',
        reason: 'uploading',
        progress: 0,
      },
    } as PendingAttachment;

    const maxSize = 10 * 1024 * 1024; // 10MB limit
    if (file.size > maxSize) {
      toast.error('File size exceeds 10MB limit');
      yield {
        id: fid,
        type: file.type.startsWith('image/') ? 'image' : 'document',
        name: file.name,
        contentType: file.type || 'application/octet-stream',
        file,
        status: {
          type: 'incomplete',
          reason: 'error',
          error: new Error('File size exceeds 10MB limit'),
        },
      } as PendingAttachment;
      return;
    }

    try {
      // 构造上传请求
      const formData = new FormData();
      formData.append('file_id', fid);
      formData.append('file', file); // 将文件加入 FormData

      const response = await fetch(`/api/config/attachments`, {
        method: 'POST',
        body: formData, // 自动设置 content-type 为 multipart/form-data
      });

      if (!response.ok) {
        throw new Error('上传失败');
      }

      // 解析响应
      const result = await response.json();
      console.log('result', result);
      if (result.code != 200) {
        throw new Error('上传失败');
      }

      // 返回成功状态
      yield {
        id: fid,
        type: file.type.startsWith('image/') ? 'image' : 'document',
        name: file.name,
        contentType: file.type || 'application/octet-stream',
        file,
        status: {
          type: 'running',
          reason: 'uploading',
          progress: 100,
        },
      } as PendingAttachment;
      return;
    } catch (error) {
      // 返回失败状态
      console.log('error', error);
      yield {
        id: fid,
        type: file.type.startsWith('image/') ? 'image' : 'document',
        name: file.name,
        contentType: file.type || 'application/octet-stream',
        file,
        status: {
          type: 'incomplete',
          reason: 'error',
          error: new Error('上传失败，请稍后重试'),
        },
      } as PendingAttachment;
      return;
    }
  }
  public async send(
    attachment: PendingAttachment,
  ): Promise<CompleteAttachment> {
    if (attachment.status.type === 'incomplete') {
      throw new Error('Attachment upload failed');
    }
    if (attachment.type === 'image') {
      const base64 = await this.fileToBase64DataURL(attachment.file);
      return {
        id: attachment.id,
        type: 'image',
        name: attachment.name,
        contentType: attachment.contentType || 'application/octet-stream',
        status: { type: 'complete' },
        content: [
          {
            type: 'image',
            image: base64, // data:image/jpeg;base64,... format
          },
        ],
      };
    } else {
      return {
        id: attachment.id,
        type: 'document',
        name: attachment.name,
        contentType: attachment.contentType || 'application/octet-stream',
        content: [],
        status: { type: 'complete' },
      };
    }
  }

  public async remove(attachment: PendingAttachment): Promise<void> {
    // Cleanup if needed
    console.log('removing attachment:', attachment);
  }

  private async fileToBase64DataURL(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        // FileReader result is already a data URL
        resolve(reader.result as string);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }
}
