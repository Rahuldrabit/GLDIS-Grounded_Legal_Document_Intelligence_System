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
  structured_fields: Array<{field: string, value: unknown, confidence: number}>;
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

export const getBYOKHeaders = (): Record<string, string> => {
  const headers: Record<string, string> = {};
  const provider = localStorage.getItem('llm_provider');
  const apiKey = localStorage.getItem('llm_api_key');
  const model = localStorage.getItem('llm_model');
  const baseUrl = localStorage.getItem('llm_base_url');
  
  if (provider) headers['X-LLM-Provider'] = provider;
  if (apiKey) headers['X-LLM-Api-Key'] = apiKey;
  if (model) headers['X-LLM-Model'] = model;
  if (baseUrl) headers['X-LLM-Base-Url'] = baseUrl;
  
  return headers;
};

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
      headers: { 
        'Content-Type': 'application/json',
        ...getBYOKHeaders()
      },
      body: JSON.stringify({ document_id: documentId, top_k: 5 }),
    });
    if (!res.ok) throw new Error('Draft generation failed');
    return res.json();
  },

  // Stateless Endpoints
  statelessVlmExtract: async (file: File): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${BASE_URL}/documents/vlm-extract`, {
      method: 'POST',
      headers: {
        ...getBYOKHeaders()
      },
      body: formData,
    });
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`Extraction failed: ${errorText}`);
    }
    return res.json();
  },

  statelessQuery: async (documentText: string, query: string): Promise<any> => {
    const res = await fetch(`${BASE_URL}/search/stateless-query`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        ...getBYOKHeaders()
      },
      body: JSON.stringify({ document_text: documentText, query, top_k: 5 }),
    });
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`Query failed: ${errorText}`);
    }
    return res.json();
  },

  statelessGenerateDraft: async (documentText: string, query: string = "Generate a case fact summary and internal memo."): Promise<any> => {
    const res = await fetch(`${BASE_URL}/drafts/stateless-generate`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        ...getBYOKHeaders()
      },
      body: JSON.stringify({ document_text: documentText, query, top_k: 5 }),
    });
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`Draft generation failed: ${errorText}`);
    }
    return res.json();
  },
};
