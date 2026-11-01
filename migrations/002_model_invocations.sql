CREATE TABLE model_invocations (
 run_id uuid NOT NULL, call_number int NOT NULL, action_id text NOT NULL,
 phase text NOT NULL CHECK(phase IN ('expand','decide')), adapter text NOT NULL, model text NOT NULL,
 status text NOT NULL CHECK(status IN ('running','completed','failed','unknown')),
 request jsonb NOT NULL, response jsonb, error jsonb,
 started_at timestamptz NOT NULL DEFAULT clock_timestamp(), finished_at timestamptz, elapsed_ms float8,
 PRIMARY KEY(run_id,call_number),
 FOREIGN KEY(run_id,action_id) REFERENCES run_actions(run_id,action_id) ON DELETE CASCADE
);
CREATE INDEX model_invocations_action ON model_invocations(run_id,action_id);
