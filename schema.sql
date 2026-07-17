-- Metadata + feature-summary store. The actual audio arrays, MFCC
-- matrices, and video segments themselves live in Nectar object storage
-- (see src/storage.py); this DB holds pointers + everything you'd want
-- to query/filter on.

CREATE TABLE IF NOT EXISTS runs (
    run_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    git_commit    TEXT,             -- git rev-parse HEAD at run time
    config_json   JSONB,            -- full config used for this run
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS videos (
    video_id      TEXT PRIMARY KEY,     -- Ego4D video_uid
    source        TEXT NOT NULL DEFAULT 'ego4d',
    duration_sec  DOUBLE PRECISION,
    swift_key     TEXT,                 -- object key for the raw video, if kept
    metadata_json JSONB,                -- raw Ego4D metadata for this video
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS segments (
    segment_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id      TEXT NOT NULL REFERENCES videos(video_id),
    parent_segment_id UUID REFERENCES segments(segment_id), -- for subgroups
    start_sec     DOUBLE PRECISION NOT NULL,
    end_sec       DOUBLE PRECISION NOT NULL,
    swift_key     TEXT,                 -- object key for extracted audio, if kept
    run_id        UUID REFERENCES runs(run_id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id);
CREATE INDEX IF NOT EXISTS idx_segments_parent ON segments(parent_segment_id);

CREATE TABLE IF NOT EXISTS features (
    feature_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    segment_id    UUID NOT NULL REFERENCES segments(segment_id),
    feature_type  TEXT NOT NULL DEFAULT 'mfcc',
    n_coeffs      INT,
    swift_key     TEXT NOT NULL,        -- object key for the .npy feature array
    summary_json  JSONB,                -- e.g. mean/var per coeff, for quick filtering
    run_id        UUID REFERENCES runs(run_id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_features_segment ON features(segment_id);
CREATE INDEX IF NOT EXISTS idx_features_type ON features(feature_type);

-- 2D projections (UMAP/t-SNE), scoped to whatever group of segments/
-- subgroups they were computed over, so you can have multiple competing
-- reductions (different params, different subgroup scopes) side by side.
CREATE TABLE IF NOT EXISTS reductions (
    reduction_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    method        TEXT NOT NULL,        -- 'umap' | 'tsne'
    params_json   JSONB,
    scope_desc    TEXT,                 -- human-readable: e.g. "all segments", "video X only"
    run_id        UUID REFERENCES runs(run_id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS reduction_points (
    reduction_id  UUID NOT NULL REFERENCES reductions(reduction_id),
    segment_id    UUID NOT NULL REFERENCES segments(segment_id),
    x             DOUBLE PRECISION NOT NULL,
    y             DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (reduction_id, segment_id)
);
