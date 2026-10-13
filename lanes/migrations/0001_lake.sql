CREATE SCHEMA IF NOT EXISTS lake_ingestion;
SET LOCAL search_path = lake_ingestion, public;

CREATE TABLE reconcile_requests (
  organization_id text PRIMARY KEY,
  generation bigint NOT NULL DEFAULT 1,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','running','completed')),
  available_at timestamptz NOT NULL DEFAULT now(),
  lease_until timestamptz,
  lease_token uuid
);
CREATE TABLE ingestion_jobs (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  lane text NOT NULL,
  sync_id bigint NOT NULL CHECK (sync_id > 0),
  fingerprint text NOT NULL,
  manual_replay boolean NOT NULL DEFAULT false,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','running','completed','failed')),
  attempts integer NOT NULL DEFAULT 0,
  available_at timestamptz NOT NULL DEFAULT now(),
  lease_until timestamptz,
  lease_token uuid,
  error_code text,
  created_at timestamptz NOT NULL DEFAULT now(),
  completed_at timestamptz,
  UNIQUE (lane, sync_id, fingerprint)
);
CREATE INDEX ingestion_jobs_pending ON ingestion_jobs (available_at) WHERE status IN ('pending','running');
CREATE TABLE ingestion_objects (
  job_id bigint NOT NULL REFERENCES ingestion_jobs(id),
  object_key text NOT NULL,
  etag text NOT NULL,
  version_id text,
  modified_at timestamptz NOT NULL,
  size_bytes bigint NOT NULL,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','running','completed','failed')),
  attempts integer NOT NULL DEFAULT 0,
  available_at timestamptz NOT NULL DEFAULT now(),
  lease_until timestamptz,
  lease_token uuid,
  error_code text,
  PRIMARY KEY (job_id, object_key, etag)
);
CREATE INDEX ingestion_objects_pending ON ingestion_objects (available_at) WHERE status IN ('pending','running');
CREATE TABLE lake_documents (
  organization_id text NOT NULL,
  lane text NOT NULL,
  source_id text NOT NULL,
  scope_value text NOT NULL,
  latest_modified_at timestamptz NOT NULL,
  latest_revision text NOT NULL,
  latest_hash text NOT NULL,
  latest_object_key text NOT NULL,
  latest_record_index bigint NOT NULL,
  desired_deleted boolean NOT NULL,
  active_revision text,
  active_hash text,
  active_object_key text,
  active_record_index bigint,
  active_fingerprint text,
  active_artifact_id text REFERENCES public."ad-spend-memory_artifact_sources"(id),
  source_uri text,
  deleted boolean NOT NULL DEFAULT false,
  PRIMARY KEY (organization_id, lane, source_id)
);
ALTER TABLE lake_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE lake_documents FORCE ROW LEVEL SECURITY;
CREATE POLICY lake_documents_tenant ON lake_documents USING (organization_id = 'org_3AP2b2tjMI6KJBA6gqLBM9HsO7g' AND organization_id = current_setting('lake.organization_id', true)) WITH CHECK (organization_id = 'org_3AP2b2tjMI6KJBA6gqLBM9HsO7g' AND organization_id = current_setting('lake.organization_id', true));

-- Existing application owners retain access. Only the new restricted runtime uses these policies.
ALTER TABLE public."ad-spend-memory_chunks" ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."ad-spend-memory_artifact_sources" ENABLE ROW LEVEL SECURITY;
CREATE POLICY lake_chunks_runtime ON public."ad-spend-memory_chunks" TO lake_runtime USING (organization_id = 'org_3AP2b2tjMI6KJBA6gqLBM9HsO7g' AND organization_id = current_setting('lake.organization_id', true)) WITH CHECK (organization_id = 'org_3AP2b2tjMI6KJBA6gqLBM9HsO7g' AND organization_id = current_setting('lake.organization_id', true));
CREATE POLICY lake_artifacts_runtime ON public."ad-spend-memory_artifact_sources" TO lake_runtime USING (organization_id = 'org_3AP2b2tjMI6KJBA6gqLBM9HsO7g' AND organization_id = current_setting('lake.organization_id', true)) WITH CHECK (organization_id = 'org_3AP2b2tjMI6KJBA6gqLBM9HsO7g' AND organization_id = current_setting('lake.organization_id', true));
CREATE INDEX memory_chunks_lake_bm25 ON public."ad-spend-memory_chunks" USING lakebase_bm25 (to_tsvector('english', child_text));
ALTER TABLE public."ad-spend-memory_chunks" SET (autovacuum_vacuum_insert_scale_factor = 0, autovacuum_vacuum_insert_threshold = 100);

GRANT USAGE ON SCHEMA lake_ingestion, public TO lake_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON reconcile_requests, ingestion_jobs, ingestion_objects, lake_documents TO lake_runtime;
GRANT USAGE, SELECT ON SEQUENCE ingestion_jobs_id_seq TO lake_runtime;
GRANT SELECT, INSERT, UPDATE ON public."ad-spend-memory_chunks", public."ad-spend-memory_artifact_sources" TO lake_runtime;
GRANT MAINTAIN ON public."ad-spend-memory_chunks" TO lake_runtime;
