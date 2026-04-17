import {
  AttachmentAdapter,
  PendingAttachment,
  CompleteAttachment,
} from '@assistant-ui/react';
import { toast } from 'sonner';

// assistant-ui tracks attachments by their client-side id during composition,
// but our backend issues the authoritative file_id during upload. We keep a
// map so that when `send()` runs for a Pending attachment we substitute in
// the server-issued id — that's what gets stored on the message and what
// the agent looks up.

// Map mime → assistant-ui's attachment type (drives how the composer
// renders the thumbnail). Backend routing uses contentType, not type, so
// this is purely a UI concern.
const typeFromMime = (mime: string): 'image' | 'file' | 'document' => {
  if (mime.startsWith('image/')) return 'image';
  if (mime.startsWith('video/') || mime.startsWith('audio/')) return 'file';
  return 'document';
};

export class UploadAttachmentAdapter implements AttachmentAdapter {
  public accept = '*/*';
  private tenantFetch: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
  private serverIdByClientId = new Map<string, string>();

  constructor(tenantFetch: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>) {
    this.tenantFetch = tenantFetch;
  }

  public async *add({
    file,
  }: {
    file: File;
  }): AsyncGenerator<PendingAttachment, void> {
    const { v4: uuidv4 } = require('uuid');
    const clientId = uuidv4();
    const contentType = file.type || 'application/octet-stream';
    const uiType = typeFromMime(contentType);

    yield {
      id: clientId,
      type: uiType,
      name: file.name,
      contentType: contentType,
      file,
      status: {
        type: 'running',
        reason: 'uploading',
        progress: 0,
      },
    } as PendingAttachment;

    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      toast.error('File size exceeds 10MB limit');
      yield {
        id: clientId,
        type: uiType,
        name: file.name,
        contentType: contentType,
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
      const formData = new FormData();
      formData.append('file', file);
      formData.append('purpose', 'chat_attachment');

      const response = await this.tenantFetch('/api/files', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      if (result.code !== 200 || !result.data?.id) {
        throw new Error(result.message || 'Upload failed');
      }

      const serverId: string = result.data.id;
      this.serverIdByClientId.set(clientId, serverId);

      yield {
        id: clientId,
        type: uiType,
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
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : 'Upload failed.';
      console.error('[upload-attachment]', error);
      toast.error(message);
      yield {
        id: clientId,
        type: uiType,
        name: file.name,
        contentType: contentType,
        file,
        status: {
          type: 'incomplete',
          reason: 'error',
          error: new Error(message),
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
    const serverId = this.serverIdByClientId.get(attachment.id) ?? attachment.id;
    return {
      id: serverId,
      type: typeFromMime(contentType),
      name: attachment.name,
      contentType: contentType,
      content: [],
      status: { type: 'complete' },
    } as CompleteAttachment;
  }

  public async remove(attachment: PendingAttachment): Promise<void> {
    // Drop the client→server id mapping. We deliberately don't DELETE the
    // server-side file here: the upload may still be referenced elsewhere
    // (other composer threads, retries) and the TTL sweep handles true
    // orphans on the schedule defined by `purpose=chat_attachment`.
    this.serverIdByClientId.delete(attachment.id);
  }
}
