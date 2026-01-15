import {
  AttachmentAdapter,
  PendingAttachment,
  CompleteAttachment,
} from '@assistant-ui/react';
import { toast } from 'sonner';

// Helper function to determine attachment type
const getAttachmentType = (mimeType: string): 'image' | 'document' | 'file' => {
  return 'document';
};

export class UploadAttachmentAdapter implements AttachmentAdapter {
  public accept = '*/*';
  private tenantFetch: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

  constructor(tenantFetch: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>) {
    this.tenantFetch = tenantFetch;
  }

  public async *add({
    file,
  }: {
    file: File;
  }): AsyncGenerator<PendingAttachment, void> {
    // Validate file size
    const { v4: uuidv4 } = require('uuid');
    const fid = uuidv4();
    const contentType = file.type || 'application/octet-stream';
    const attachmentType = getAttachmentType(contentType);
    
    yield {
      id: fid,
      type: attachmentType,
      name: file.name,
      contentType: contentType,
      file,
      status: {
        type: 'running',
        reason: 'uploading',
        progress: 0,
      },
    } as PendingAttachment;

    const maxSize = 10 * 1024 * 1024; // 10MB limit
    
    if (file.size > maxSize) {
      toast.error(`File size exceeds 10MB limit`);
      yield {
        id: fid,
        type: attachmentType,
        name: file.name,
        contentType: contentType,
        file,
        status: {
          type: 'incomplete',
          reason: 'error',
          error: new Error(`File size exceeds 10MB limit`),
        },
      } as PendingAttachment;
      return;
    }

    try {
      // 构造上传请求
      const formData = new FormData();
      formData.append('file_id', fid);
      formData.append('file', file); // 将文件加入 FormData

      const response = await this.tenantFetch(`/api/config/attachments`, {
        method: 'POST',
        body: formData, // 自动设置 content-type 为 multipart/form-data
      });

      // 解析响应
      const result = await response.json();
      console.log('result', result);
      if (result.code !== 200) {
        throw new Error(result.message);
      }

      // 返回成功状态
      yield {
        id: fid,
        type: attachmentType,
        name: file.name,
        contentType: contentType,
        file,
        status: {
          type: 'running',
          reason: 'uploading',
          progress: 100,
        },
      } as PendingAttachment;
      return;
    } catch (error: any) {
      // 返回失败状态
      console.log('error', error);
      toast.error(error.message || '上传失败，请稍后重试');
      yield {
        id: fid,
        type: attachmentType,
        name: file.name,
        contentType: contentType,
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
    
    const contentType = attachment.contentType || 'application/octet-stream';
    console.log("upload attachment success:", attachment);
    return {
      id: attachment.id,
      type: 'document',
      name: attachment.name,
      contentType: contentType,
      content: [],
      status: { type: 'complete' },
    } as CompleteAttachment;
  }

  public async remove(attachment: PendingAttachment): Promise<void> {
    // Cleanup if needed
    console.log('removing attachment:', attachment);
  }

}
