export type KB = {
  id: string;
  name: string;
  current_version_id: string | null;
};
export type Version = {
  id: string;
  status: string;
  published_at: string | null;
  base_version_id: string | null;
  created_at: string;
  error?: { message: string };
  config: { chunking: ChunkConfig };
};
export type ChunkConfig = {
  target_tokens: number;
  max_tokens: number;
  overlap_tokens: number;
  semantic_threshold: number;
};
export type Doc = {
  document_id: string;
  document_version_id: string;
  metadata: { title: string };
  version_date: string;
  name: string;
};
export type Citation = {
  evidence_id: string;
  kb_version_id: string;
  document_id: string;
  document_version_id: string;
  quote: string;
  version_date: string;
  locator: Record<string, unknown>[];
};
export type Result = {
  answer: string;
  task_status: string;
  citations: Citation[];
  information_gaps: string[];
  questions?: string[];
  deadline_at?: string;
};
export type Run = {
  id: string;
  question: string;
  execution_status: string;
  task_status: string | null;
  result: Result | null;
  deadline_at: string | null;
  kb_version_id: string;
};
export type Trace = {
  seq: number;
  type: string;
  occurred_at: string;
  payload: Record<string, unknown>;
};
export type Job = {
  id: string;
  status: string;
  error?: { code: string; message: string };
  progress?: {
    stage: string;
    completed?: number;
    total?: number;
    published?: boolean;
    publish_error?: { message: string };
  };
};
export type Pending = {
  upload_id?: string;
  document_id: string;
  name: string;
  action: "add" | "replace" | "remove";
};
