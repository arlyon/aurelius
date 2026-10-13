CREATE EXTENSION vector;
CREATE SCHEMA lake_ingestion;
DO $$ BEGIN CREATE ROLE lake_runtime NOSUPERUSER NOBYPASSRLS; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
GRANT USAGE ON SCHEMA lake_ingestion,public TO lake_runtime;
CREATE TABLE public."ad-spend-memory_artifact_sources" (
 id text PRIMARY KEY, organization_id text NOT NULL, source_type text NOT NULL, source_uri text NOT NULL,
 external_id text NOT NULL, revision_id text, content_hash text, mime_type text, metadata jsonb, deleted_at timestamp,
 created_at timestamp NOT NULL DEFAULT now(), updated_at timestamp NOT NULL DEFAULT now()
);
CREATE TABLE public."ad-spend-memory_chunks" (
 id text PRIMARY KEY, organization_id text NOT NULL, artifact_source_id text NOT NULL REFERENCES public."ad-spend-memory_artifact_sources"(id),
 source_path text NOT NULL, source_hash text, chunk_index integer NOT NULL, context_window_id text NOT NULL,
 child_text text NOT NULL, parent_block text NOT NULL, context_header text NOT NULL, metadata jsonb, embedding vector(1280) NOT NULL,
 created_at timestamp NOT NULL DEFAULT now(), updated_at timestamp NOT NULL DEFAULT now()
);
CREATE TABLE lake_ingestion.lake_documents (
 organization_id text NOT NULL, lane text NOT NULL, source_id text NOT NULL, scope_value text NOT NULL,
 latest_modified_at timestamptz NOT NULL, latest_revision text NOT NULL, latest_hash text NOT NULL, latest_object_key text NOT NULL,
 latest_record_index bigint NOT NULL, desired_deleted boolean NOT NULL,
 active_revision text, active_hash text, active_object_key text, active_record_index bigint, active_fingerprint text,
 active_artifact_id text REFERENCES public."ad-spend-memory_artifact_sources"(id), source_uri text, deleted boolean NOT NULL DEFAULT false,
 PRIMARY KEY(organization_id,lane,source_id)
);
CREATE TABLE lake_ingestion.ingestion_jobs (id bigserial PRIMARY KEY,lane text NOT NULL,status text NOT NULL);
CREATE TABLE lake_ingestion.ingestion_objects (
 job_id bigint NOT NULL REFERENCES lake_ingestion.ingestion_jobs(id),object_key text NOT NULL,etag text NOT NULL,
 lease_token uuid NOT NULL,status text NOT NULL,lease_until timestamptz NOT NULL,
 PRIMARY KEY(job_id,object_key,etag)
);
-- Substrate for the changed writer/reader contracts. BM25 retrieval and the model
-- runtime are intentionally separate from this change-feed integration suite.
