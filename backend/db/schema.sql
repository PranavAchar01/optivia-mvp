-- Optivia Trace Contract — §6.4
-- Run once against your Postgres 16 + pgvector instance.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ---------------------------------------------------------------------------
-- traces — one row per Optivia turn (the core data moat)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS traces (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL,
    workspace_id        UUID NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW(),

    -- Raw input
    raw_prompt          TEXT NOT NULL,
    raw_prompt_hash     BYTEA NOT NULL,
    raw_prompt_emb      VECTOR(1024),          -- voyage-code-3 embedding

    -- Project context snapshot
    project_context     JSONB NOT NULL DEFAULT '{}',
    taxonomy_version    TEXT NOT NULL DEFAULT 'v1',

    -- Pipeline outputs (the "trace contract" moat)
    fast_intent         JSONB,
    classification      JSONB,
    scores              JSONB,
    clarifications      JSONB,
    master_prompt       TEXT,
    workflow_plan       JSONB,
    routing_decision    JSONB,
    shadow_routes       JSONB,                 -- Stage 2: all router decisions logged
    outcome             JSONB,
    feedback            JSONB,

    -- Cost / telemetry
    cost_usd            NUMERIC(10,6) DEFAULT 0,
    tokens_in           INT,
    tokens_out          INT,
    wall_ms             INT,
    retry_count         INT DEFAULT 0,
    langfuse_trace_id   TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS traces_emb_idx
    ON traces USING ivfflat (raw_prompt_emb vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS traces_class_idx
    ON traces ((classification->>'task_type'));

CREATE INDEX IF NOT EXISTS traces_user_idx
    ON traces (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS traces_hash_idx
    ON traces (raw_prompt_hash);

CREATE INDEX IF NOT EXISTS traces_workspace_idx
    ON traces (workspace_id, created_at DESC);


-- ---------------------------------------------------------------------------
-- routing_decisions — logs all router alternatives (needed for Stage 2 training)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS routing_decisions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id        UUID REFERENCES traces(id) ON DELETE CASCADE,
    router_name     TEXT NOT NULL,
    was_active      BOOLEAN NOT NULL DEFAULT TRUE,
    chosen_model    TEXT,
    alternatives    JSONB,
    router_features JSONB,
    router_score    FLOAT
);

CREATE INDEX IF NOT EXISTS routing_decisions_trace_idx
    ON routing_decisions (trace_id);


-- ---------------------------------------------------------------------------
-- outcomes — execution results (join with traces for training labels)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS outcomes (
    trace_id                UUID PRIMARY KEY REFERENCES traces(id) ON DELETE CASCADE,
    exit_code               INT,
    diff_lines_added        INT,
    diff_lines_removed      INT,
    files_touched           INT,
    tests_passed            BOOLEAN,
    user_accepted           BOOLEAN,
    user_thumbs             SMALLINT,          -- -1 / 0 / 1
    followup_prompt         TEXT,
    followup_within_seconds INT
);


-- ---------------------------------------------------------------------------
-- sessions — multi-turn session state (§10.1)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sessions (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                 UUID NOT NULL,
    workspace_id            UUID NOT NULL,
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW(),
    original_prompt         TEXT,
    cumulative_context      JSONB DEFAULT '{}',
    token_budget_consumed   INT DEFAULT 0,
    q_history               JSONB DEFAULT '[]',  -- sliding window of Q_t scores
    conflict_log            JSONB DEFAULT '[]',
    fleet_state             JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS sessions_user_idx
    ON sessions (user_id, updated_at DESC);


-- ---------------------------------------------------------------------------
-- workspaces — per-tenant budget + config
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- experience_records — §5.3.3 / §5.3.19: ExpeL-inspired memory pool
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS experience_records (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          TIMESTAMPTZ DEFAULT NOW(),

    -- Scope + ownership (§5.3.3 cross-scope conflict resolution)
    scope               TEXT NOT NULL DEFAULT 'project',  -- project | user | global
    workspace_id        UUID,
    user_id             UUID,

    -- Task fingerprint for retrieval scoring
    task_type           TEXT NOT NULL,
    tags                TEXT[] DEFAULT '{}',

    -- Lesson content
    lesson              TEXT NOT NULL,
    lesson_emb          VECTOR(1024),
    failure_modes       JSONB DEFAULT '[]',
    successful_patterns JSONB DEFAULT '[]',
    outcome_label       TEXT NOT NULL DEFAULT 'success',  -- success | failure

    -- ExpeL operator state
    weight              INT NOT NULL DEFAULT 2,
    trust_score         FLOAT NOT NULL DEFAULT 1.0,
    conf_count          INT NOT NULL DEFAULT 1,

    -- Staleness metadata
    last_confirmed_run  INT NOT NULL DEFAULT 0,
    last_confirmed      TIMESTAMPTZ DEFAULT NOW(),
    archived            BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS exp_scope_ws_idx
    ON experience_records (scope, workspace_id, task_type)
    WHERE archived = FALSE;

CREATE INDEX IF NOT EXISTS exp_emb_idx
    ON experience_records USING ivfflat (lesson_emb vector_cosine_ops)
    WITH (lists = 50);

CREATE INDEX IF NOT EXISTS exp_tags_idx
    ON experience_records USING GIN (tags);


-- ---------------------------------------------------------------------------
-- bandit_state — §5.3.12: D-LinUCB per-arm posterior, persisted
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS bandit_state (
    workspace_id    UUID NOT NULL,
    arm_id          TEXT NOT NULL,           -- A0..A5
    d               INT NOT NULL,            -- context dim
    a_matrix        JSONB NOT NULL,          -- d×d matrix (list-of-lists)
    b_vector        JSONB NOT NULL,          -- d-vector
    n_observations  INT NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (workspace_id, arm_id)
);


CREATE TABLE IF NOT EXISTS workspaces (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                TEXT NOT NULL,
    owner_user_id       UUID NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    monthly_budget_usd  NUMERIC(10,2) DEFAULT 50.0,
    spend_this_month    NUMERIC(10,2) DEFAULT 0.0,
    config              JSONB DEFAULT '{}'        -- per-tenant threshold overrides
);
