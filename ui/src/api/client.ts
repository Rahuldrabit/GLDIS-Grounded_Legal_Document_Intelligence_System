const BASE_URL = 'http://localhost:8000/api';

export interface DocumentListResponse {
  document_id: string;
  filename: string;
  status: string;
  page_count: number;
  upload_time: string;
}

export interface DocumentDetailResponse extends DocumentListResponse {
  file_size: number;
  chunks_count: number;
  processed_time?: string;
  error_message?: string;
  structured_fields: Array<{field: string, value: any, confidence: number}>;
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  status: string;
  message: string;
}

export interface ProcessResponse {
  document_id: string;
  status: string;
  chunks_created: number;
  entities_extracted: number;
  message: string;
}

export interface DraftResponse {
  draft_id: string;
  document_id: string;
  generated_text: string;
  grounding_score: number;
}

export const api = {
  uploadDocument: async (file: File): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${BASE_URL}/documents/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Upload failed');
    return res.json();
  },
  
  processDocumentSync: async (documentId: string): Promise<ProcessResponse> => {
    const res = await fetch(`${BASE_URL}/documents/${documentId}/process/sync`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Process failed');
    return res.json();
  },

  listDocuments: async (): Promise<DocumentListResponse[]> => {
    const res = await fetch(`${BASE_URL}/documents`);
    if (!res.ok) throw new Error('Failed to fetch documents');
    return res.json();
  },

  getDocument: async (id: string): Promise<DocumentDetailResponse> => {
    const res = await fetch(`${BASE_URL}/documents/${id}`);
    if (!res.ok) throw new Error('Failed to fetch document');
    return res.json();
  },

  deleteDocument: async (id: string): Promise<void> => {
    const res = await fetch(`${BASE_URL}/documents/${id}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete document');
  },

  generateDraft: async (documentId: string): Promise<DraftResponse> => {
    const res = await fetch(`${BASE_URL}/drafts/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ document_id: documentId, top_k: 5 }),
    });
    if (!res.ok) throw new Error('Draft generation failed');
    return res.json();
  },
};
