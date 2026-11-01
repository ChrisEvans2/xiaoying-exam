CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS schema_migrations (version text PRIMARY KEY, applied_at timestamptz DEFAULT now());
CREATE TABLE knowledge_bases (
 id uuid PRIMARY KEY, name text NOT NULL, current_version_id uuid, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE kb_versions (
 id uuid PRIMARY KEY, kb_id uuid NOT NULL REFERENCES knowledge_bases, base_version_id uuid,
 status text NOT NULL CHECK(status IN ('pending','building','ready','failed')),
 config jsonb NOT NULL, manifest_hash text NOT NULL, error jsonb, published_at timestamptz,
 created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(id,kb_id),
 FOREIGN KEY(base_version_id,kb_id) REFERENCES kb_versions(id,kb_id)
);
ALTER TABLE knowledge_bases ADD CONSTRAINT current_version_fk
 FOREIGN KEY(current_version_id,id) REFERENCES kb_versions(id,kb_id);
CREATE TABLE documents (
 id uuid PRIMARY KEY, kb_id uuid NOT NULL REFERENCES knowledge_bases, document_id text NOT NULL,
 UNIQUE(kb_id,document_id), UNIQUE(id,kb_id)
);
CREATE TABLE uploads (
 id uuid PRIMARY KEY, kb_id uuid NOT NULL REFERENCES knowledge_bases, name text NOT NULL,
 file_hash text NOT NULL, storage_path text NOT NULL, metadata jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(id,kb_id)
);
CREATE TABLE document_versions (
 id uuid PRIMARY KEY, document_fk uuid NOT NULL, kb_id uuid NOT NULL, upload_id uuid NOT NULL,
 metadata jsonb NOT NULL, version_date date NOT NULL,
 FOREIGN KEY(document_fk,kb_id) REFERENCES documents(id,kb_id),
 FOREIGN KEY(upload_id,kb_id) REFERENCES uploads(id,kb_id),
 UNIQUE(id,document_fk,kb_id), UNIQUE(id,kb_id)
);
CREATE TABLE kb_version_documents (
 kb_version_id uuid NOT NULL, kb_id uuid NOT NULL, document_fk uuid NOT NULL, document_version_id uuid NOT NULL,
 PRIMARY KEY(kb_version_id,document_fk), UNIQUE(kb_version_id,document_version_id),
 FOREIGN KEY(kb_version_id,kb_id) REFERENCES kb_versions(id,kb_id),
 FOREIGN KEY(document_version_id,document_fk,kb_id) REFERENCES document_versions(id,document_fk,kb_id)
);
CREATE TABLE source_blocks (
 kb_version_id uuid NOT NULL, document_version_id uuid NOT NULL, block_id text NOT NULL,
 text text NOT NULL, locator jsonb NOT NULL, block_type text NOT NULL, ordinal int NOT NULL,
 PRIMARY KEY(kb_version_id,document_version_id,block_id),
 FOREIGN KEY(kb_version_id,document_version_id) REFERENCES kb_version_documents(kb_version_id,document_version_id)
);
CREATE TABLE chunks (
 id uuid PRIMARY KEY, kb_version_id uuid NOT NULL, document_version_id uuid NOT NULL,
 ordinal int NOT NULL, text text NOT NULL, source_ranges jsonb NOT NULL, heading_path jsonb NOT NULL,
 search_vector tsvector NOT NULL, embedding vector(1024) NOT NULL, token_count int NOT NULL CHECK(token_count>0),
 UNIQUE(kb_version_id,document_version_id,ordinal), UNIQUE(id,kb_version_id,document_version_id),
 FOREIGN KEY(kb_version_id,document_version_id) REFERENCES kb_version_documents(kb_version_id,document_version_id)
);
CREATE INDEX chunks_version ON chunks(kb_version_id,document_version_id);
CREATE INDEX chunks_fts ON chunks USING gin(search_vector);
CREATE TABLE conversations (
 id uuid PRIMARY KEY, kb_id uuid NOT NULL REFERENCES knowledge_bases, created_at timestamptz DEFAULT now(),
 UNIQUE(id,kb_id)
);
CREATE TABLE runs (
 id uuid PRIMARY KEY, conversation_id uuid NOT NULL, kb_id uuid NOT NULL, kb_version_id uuid NOT NULL,
 question text NOT NULL, config jsonb NOT NULL, execution_status text NOT NULL DEFAULT 'queued'
 CHECK(execution_status IN ('queued','running','waiting','finished')),
 task_status text CHECK(task_status IN ('answered','awaiting_user','incomplete','insufficient')),
 result jsonb, evidence_epoch int NOT NULL DEFAULT 0, clarification_count int NOT NULL DEFAULT 0
 CHECK(clarification_count BETWEEN 0 AND 1), deadline_at timestamptz,
 model_calls int NOT NULL DEFAULT 0, tool_calls int NOT NULL DEFAULT 0, graph_steps int NOT NULL DEFAULT 0,
 active_ms float8 NOT NULL DEFAULT 0, segment_started_at timestamptz,
 next_event_seq bigint NOT NULL DEFAULT 0, created_at timestamptz NOT NULL DEFAULT now(),
 FOREIGN KEY(conversation_id,kb_id) REFERENCES conversations(id,kb_id),
 FOREIGN KEY(kb_version_id,kb_id) REFERENCES kb_versions(id,kb_id), UNIQUE(id,kb_version_id)
);
CREATE UNIQUE INDEX one_active_run ON runs(conversation_id) WHERE execution_status IN ('queued','running','waiting');
CREATE INDEX waiting_deadline ON runs(deadline_at) WHERE execution_status='waiting';
CREATE TABLE messages (
 id bigserial PRIMARY KEY, conversation_id uuid NOT NULL REFERENCES conversations, run_id uuid NOT NULL REFERENCES runs,
 role text NOT NULL, content jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE clarifications (
 run_id uuid PRIMARY KEY REFERENCES runs, payload jsonb NOT NULL, deadline_at timestamptz NOT NULL,
 status text NOT NULL CHECK(status IN ('pending','submitted','abandoned','expired')),
 response jsonb, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE jobs (
 id uuid PRIMARY KEY, type text NOT NULL CHECK(type IN ('build','run')), resource_id uuid NOT NULL,
 payload jsonb NOT NULL DEFAULT '{}', status text NOT NULL DEFAULT 'pending'
 CHECK(status IN ('pending','running','done','failed')),
 attempts int NOT NULL DEFAULT 0, max_attempts int NOT NULL DEFAULT 3,
 next_attempt_at timestamptz NOT NULL DEFAULT now(), lease_until timestamptz,
 lease_token bigint NOT NULL DEFAULT 0, worker_id text, progress jsonb, error jsonb,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX one_active_job ON jobs(type,resource_id) WHERE status IN ('pending','running');
CREATE INDEX job_queue ON jobs(type,next_attempt_at,created_at) WHERE status='pending';
CREATE TABLE run_actions (
 run_id uuid NOT NULL REFERENCES runs, action_id text NOT NULL, kind text NOT NULL, input_hash text NOT NULL,
 status text NOT NULL CHECK(status IN ('reserved','done','unknown')), output jsonb,
 created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(run_id,action_id)
);
CREATE TABLE evidence_reads (
 id uuid PRIMARY KEY, run_id uuid NOT NULL REFERENCES runs, evidence_epoch int NOT NULL,
 kb_version_id uuid NOT NULL, document_version_id uuid NOT NULL, chunk_id uuid NOT NULL,
 payload jsonb NOT NULL, FOREIGN KEY(chunk_id,kb_version_id,document_version_id)
 REFERENCES chunks(id,kb_version_id,document_version_id)
);
CREATE TABLE trace_events (
 run_id uuid NOT NULL REFERENCES runs, seq bigint NOT NULL, event_id text NOT NULL, type text NOT NULL,
 payload jsonb NOT NULL, occurred_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(run_id,seq), UNIQUE(run_id,event_id)
);
CREATE TABLE idempotency_requests (
 scope text NOT NULL, key text NOT NULL, request_hash text NOT NULL, response jsonb,
 PRIMARY KEY(scope,key), created_at timestamptz NOT NULL DEFAULT now()
);
