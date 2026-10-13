-- Only this tenant's writer allocates cursors under an advisory transaction lock.
-- Sequence order is therefore commit order per tenant, including concurrent object workers.
-- Migration 0001 intentionally restricts its forced-RLS ledger to this canary.
SELECT set_config('lake.organization_id','org_3AP2b2tjMI6KJBA6gqLBM9HsO7g',true);
ALTER TABLE lake_ingestion.lake_documents ADD COLUMN latest_source_modified_at timestamptz;

CREATE TABLE lake_ingestion.lanes (
  organization_id text NOT NULL,
  lane text NOT NULL,
  fingerprint text NOT NULL,
  enabled boolean NOT NULL,
  PRIMARY KEY (organization_id,lane)
);
ALTER TABLE lake_ingestion.lanes ENABLE ROW LEVEL SECURITY;
ALTER TABLE lake_ingestion.lanes FORCE ROW LEVEL SECURITY;
CREATE POLICY lanes_tenant ON lake_ingestion.lanes USING (organization_id = current_setting('lake.organization_id', true)) WITH CHECK (organization_id = current_setting('lake.organization_id', true));
GRANT SELECT, INSERT, UPDATE ON lake_ingestion.lanes TO lake_runtime;
INSERT INTO lake_ingestion.lanes SELECT DISTINCT ON (organization_id,lane) organization_id,lane,active_fingerprint,true FROM lake_ingestion.lake_documents WHERE active_fingerprint IS NOT NULL ORDER BY organization_id,lane,latest_modified_at DESC;

CREATE TABLE lake_ingestion.document_changes (
  cursor bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  organization_id text NOT NULL,
  lane text NOT NULL,
  source_id text NOT NULL,
  artifact_source_id text NOT NULL REFERENCES public."ad-spend-memory_artifact_sources"(id),
  revision text NOT NULL,
  content_hash text NOT NULL,
  indexed_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  extracted_at timestamptz NOT NULL,
  source_modified_at timestamptz,
  source_uri text NOT NULL,
  deleted boolean NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX document_changes_tenant_cursor ON lake_ingestion.document_changes (organization_id, cursor);
ALTER TABLE lake_ingestion.document_changes ENABLE ROW LEVEL SECURITY;
ALTER TABLE lake_ingestion.document_changes FORCE ROW LEVEL SECURITY;
CREATE POLICY document_changes_tenant ON lake_ingestion.document_changes
  USING (organization_id = current_setting('lake.organization_id', true))
  WITH CHECK (organization_id = current_setting('lake.organization_id', true));
GRANT SELECT, INSERT ON lake_ingestion.document_changes TO lake_runtime;
GRANT USAGE, SELECT ON SEQUENCE lake_ingestion.document_changes_cursor_seq TO lake_runtime;

-- The initial corpus is already committed. A quiesced migration backfills its
-- active revisions so a new review consumer can start at cursor zero.
INSERT INTO lake_ingestion.document_changes
  (organization_id,lane,source_id,artifact_source_id,revision,content_hash,extracted_at,source_uri,deleted,metadata)
SELECT organization_id,lane,source_id,active_artifact_id,active_revision,active_hash,latest_modified_at,
  coalesce(source_uri,''),deleted OR desired_deleted,
  jsonb_build_object('backfill',true,'source_type',CASE WHEN lane LIKE '%slack%' THEN 'slack' ELSE 'document' END)
FROM lake_ingestion.lake_documents WHERE active_artifact_id IS NOT NULL
ORDER BY organization_id,lane,source_id;
