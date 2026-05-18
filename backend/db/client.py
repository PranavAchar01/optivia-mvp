"""Async Postgres client (asyncpg) for trace writes and reads."""

from __future__ import annotations

import json
from typing import Any

import asyncpg
import structlog

from backend.config import settings

log = structlog.get_logger(__name__)


def _pgvector_literal(values: list[float]) -> str:
    """Format a Python list as a pgvector text literal: '[v1,v2,...]'."""
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


class DBClient:
    _pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
        self._pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
        log.info("db.connected")

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()

    async def insert_trace(self, record: dict[str, Any]) -> None:
        if not self._pool:
            log.warning("db.not_connected — trace not persisted")
            return

        emb = record.get("raw_prompt_emb")
        emb_literal = _pgvector_literal(emb) if emb else None

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO traces (
                    id, user_id, workspace_id,
                    raw_prompt, raw_prompt_hash, raw_prompt_emb,
                    project_context, taxonomy_version,
                    fast_intent, classification, scores, clarifications,
                    master_prompt, workflow_plan, routing_decision,
                    outcome, feedback,
                    cost_usd, tokens_in, tokens_out, wall_ms, retry_count
                ) VALUES (
                    $1, $2, $3,
                    $4, $5, $6::vector,
                    $7, $8,
                    $9, $10, $11, $12,
                    $13, $14, $15,
                    $16, $17,
                    $18, $19, $20, $21, $22
                )
                ON CONFLICT (id) DO NOTHING
                """,
                record.get("id"),
                record.get("user_id") or "00000000-0000-0000-0000-000000000000",
                record.get("workspace_id") or "00000000-0000-0000-0000-000000000000",
                record.get("raw_prompt", ""),
                record.get("raw_prompt_hash", b""),
                emb_literal,
                json.dumps(record.get("project_context", {})),
                record.get("taxonomy_version", "v1"),
                json.dumps(record.get("fast_intent", {})),
                json.dumps(record.get("classification", {})),
                json.dumps(record.get("scores", {})),
                json.dumps(record.get("clarifications", {})),
                record.get("master_prompt", ""),
                json.dumps(record.get("workflow_plan", {})),
                json.dumps(record.get("routing_decision", {})),
                json.dumps(record.get("outcome", {})),
                json.dumps(record.get("feedback", {})),
                record.get("cost_usd", 0.0),
                record.get("tokens_in", 0),
                record.get("tokens_out", 0),
                record.get("wall_ms", 0),
                record.get("retry_count", 0),
            )

    async def find_semantic_cache(
        self,
        embedding: list[float],
        workspace_id: str,
        threshold: float = 0.95,
        same_task_type: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Tier 1 semantic cache: returns the most-similar trace whose cosine
        similarity to `embedding` is at least `threshold`, scoped to the
        same workspace and (optionally) the same task type (§4.5).
        """
        if not self._pool or not embedding:
            return None
        emb_literal = _pgvector_literal(embedding)
        async with self._pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    """
                    SELECT
                        id, master_prompt, workflow_plan, routing_decision,
                        1 - (raw_prompt_emb <=> $1::vector) AS similarity
                    FROM traces
                    WHERE workspace_id = $2::uuid
                      AND raw_prompt_emb IS NOT NULL
                      AND ($3::text IS NULL OR classification->>'task_type' = $3)
                      AND (outcome->>'user_accepted')::boolean IS NOT FALSE
                    ORDER BY raw_prompt_emb <=> $1::vector
                    LIMIT 1
                    """,
                    emb_literal,
                    workspace_id,
                    same_task_type,
                )
                if row and row["similarity"] is not None and row["similarity"] >= threshold:
                    return dict(row)
            except Exception as exc:
                log.warning("db.semantic_cache_error", error=str(exc))
        return None

    async def similar_exemplars(
        self,
        embedding: list[float],
        workspace_id: str,
        task_type: str | None = None,
        k: int = 3,
    ) -> list[dict[str, Any]]:
        """Stage-2 retrieval: top-k historically successful master prompts (§5.7)."""
        if not self._pool or not embedding:
            return []
        emb_literal = _pgvector_literal(embedding)
        async with self._pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    """
                    SELECT
                        id, raw_prompt, master_prompt,
                        1 - (raw_prompt_emb <=> $1::vector) AS similarity
                    FROM traces
                    WHERE workspace_id = $2::uuid
                      AND raw_prompt_emb IS NOT NULL
                      AND ($3::text IS NULL OR classification->>'task_type' = $3)
                      AND (outcome->>'user_accepted')::boolean IS TRUE
                    ORDER BY raw_prompt_emb <=> $1::vector
                    LIMIT $4
                    """,
                    emb_literal,
                    workspace_id,
                    task_type,
                    k,
                )
                return [dict(r) for r in rows]
            except Exception:
                return []

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        if not self._pool:
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM traces WHERE id = $1", trace_id)
            return dict(row) if row else None

    async def get_session_traces(self, workspace_id: str, limit: int = 50) -> list[dict[str, Any]]:
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, user_id, workspace_id, created_at,
                       raw_prompt, project_context, taxonomy_version,
                       fast_intent, classification, scores, clarifications,
                       master_prompt, workflow_plan, routing_decision,
                       outcome, feedback,
                       cost_usd, tokens_in, tokens_out, wall_ms, retry_count,
                       langfuse_trace_id
                FROM traces
                WHERE workspace_id = $1::uuid
                ORDER BY created_at DESC
                LIMIT $2
                """,
                workspace_id, limit,
            )
            result = []
            for r in rows:
                d = dict(r)
                for k in ("project_context", "fast_intent", "classification", "scores",
                          "clarifications", "workflow_plan", "routing_decision",
                          "outcome", "feedback"):
                    v = d.get(k)
                    if isinstance(v, str):
                        try:
                            d[k] = json.loads(v)
                        except (json.JSONDecodeError, TypeError):
                            pass
                if d.get("cost_usd") is not None:
                    d["cost_usd"] = float(d["cost_usd"])
                result.append(d)
            return result

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        if not self._pool:
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
            return dict(row) if row else None

    async def insert_routing_decisions(
        self,
        trace_id: str,
        rows: list[dict[str, Any]],
    ) -> None:
        """Persists every routing decision (active + shadow) per §4.4."""
        if not self._pool or not rows:
            return
        async with self._pool.acquire() as conn:
            for row in rows:
                try:
                    await conn.execute(
                        """
                        INSERT INTO routing_decisions (
                            id, trace_id, router_name, was_active,
                            chosen_model, alternatives, router_features, router_score
                        ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        row.get("id"),
                        trace_id,
                        row["router_name"],
                        bool(row.get("was_active", False)),
                        row.get("chosen_model"),
                        json.dumps(row.get("alternatives", [])),
                        json.dumps(row.get("router_features", {})),
                        float(row.get("router_score", 0.0)),
                    )
                except Exception as exc:
                    log.warning("db.routing_decision_insert_error", error=str(exc))

    async def insert_outcome(self, trace_id: str, outcome: dict[str, Any]) -> None:
        """Writes a normalised outcomes row for Stage 2 training labels (§4.4, §5.4)."""
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO outcomes (
                        trace_id, exit_code, diff_lines_added, diff_lines_removed,
                        files_touched, tests_passed, user_accepted, user_thumbs,
                        followup_prompt, followup_within_seconds
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (trace_id) DO UPDATE SET
                        exit_code = EXCLUDED.exit_code,
                        diff_lines_added = EXCLUDED.diff_lines_added,
                        diff_lines_removed = EXCLUDED.diff_lines_removed,
                        files_touched = EXCLUDED.files_touched,
                        tests_passed = EXCLUDED.tests_passed,
                        user_accepted = EXCLUDED.user_accepted,
                        user_thumbs = EXCLUDED.user_thumbs,
                        followup_prompt = EXCLUDED.followup_prompt,
                        followup_within_seconds = EXCLUDED.followup_within_seconds
                    """,
                    trace_id,
                    outcome.get("exit_code"),
                    outcome.get("diff_lines_added"),
                    outcome.get("diff_lines_removed"),
                    outcome.get("files_touched"),
                    outcome.get("tests_passed"),
                    outcome.get("user_accepted"),
                    outcome.get("user_thumbs"),
                    outcome.get("followup_prompt"),
                    outcome.get("followup_within_seconds"),
                )
            except Exception as exc:
                log.warning("db.outcome_insert_error", error=str(exc))

    async def update_trace_feedback(self, trace_id: str, feedback: dict[str, Any]) -> None:
        """Updates the feedback JSONB column on a trace + the outcomes row."""
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    UPDATE traces
                    SET feedback = $1::jsonb
                    WHERE id = $2
                    """,
                    json.dumps(feedback),
                    trace_id,
                )
                # Also reflect into outcomes for the training join
                await conn.execute(
                    """
                    UPDATE outcomes
                    SET user_thumbs = $1,
                        user_accepted = $2,
                        followup_prompt = $3
                    WHERE trace_id = $4
                    """,
                    feedback.get("thumbs"),
                    (feedback.get("thumbs") or 0) > 0,
                    feedback.get("followup_prompt"),
                    trace_id,
                )
            except Exception as exc:
                log.warning("db.feedback_update_error", error=str(exc))

    async def aggregate_metrics(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Aggregate evaluation metrics used by the nightly eval job (§4.10)."""
        if not self._pool:
            return {}
        async with self._pool.acquire() as conn:
            where = "WHERE workspace_id = $1::uuid" if workspace_id else ""
            params = [workspace_id] if workspace_id else []
            try:
                summary = await conn.fetchrow(
                    f"""
                    SELECT
                        COUNT(*)::int AS total_traces,
                        AVG((scores->>'complexity')::float)::float AS avg_complexity,
                        AVG((scores->>'specificity')::float)::float AS avg_specificity,
                        AVG(wall_ms)::float AS avg_wall_ms,
                        SUM(tokens_in)::int AS total_tokens_in,
                        SUM(tokens_out)::int AS total_tokens_out
                    FROM traces
                    {where}
                    """,
                    *params,
                )
                router_counts = await conn.fetch(
                    f"""
                    SELECT chosen_model, was_active, COUNT(*)::int AS n
                    FROM routing_decisions rd
                    JOIN traces t ON t.id = rd.trace_id
                    {where.replace('workspace_id', 't.workspace_id') if where else ''}
                    GROUP BY chosen_model, was_active
                    ORDER BY n DESC
                    """,
                    *params,
                )
                return {
                    "summary": dict(summary) if summary else {},
                    "router_distribution": [dict(r) for r in router_counts],
                }
            except Exception as exc:
                log.warning("db.aggregate_metrics_error", error=str(exc))
                return {}

    async def upsert_session(
        self,
        session_id: str,
        user_id: str,
        workspace_id: str,
        original_prompt: str,
        cumulative_delta: dict[str, Any],
        token_budget_consumed: int,
        q_score: float,
        fleet_state: dict[str, Any],
    ) -> None:
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (
                    id, user_id, workspace_id, original_prompt,
                    cumulative_context, token_budget_consumed, q_history, fleet_state
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
                ON CONFLICT (id) DO UPDATE SET
                    updated_at = NOW(),
                    cumulative_context = sessions.cumulative_context || $5::jsonb,
                    token_budget_consumed = sessions.token_budget_consumed + $6,
                    q_history = (
                        SELECT jsonb_agg(val) FROM (
                            SELECT val FROM jsonb_array_elements(sessions.q_history) val
                            UNION ALL SELECT to_jsonb($9::float)
                            LIMIT 20
                        ) sub
                    ),
                    fleet_state = $8
                """,
                session_id,
                user_id or "00000000-0000-0000-0000-000000000000",
                workspace_id or "00000000-0000-0000-0000-000000000000",
                original_prompt,
                json.dumps(cumulative_delta),
                token_budget_consumed,
                json.dumps([q_score]),
                json.dumps(fleet_state),
                q_score,
            )


    # ────────────────────────────────────────────────────────────────────────
    # Experience memory (§5.3.3, §5.3.19)
    # ────────────────────────────────────────────────────────────────────────

    async def get_experiences_for_retrieval(
        self,
        workspace_id: str,
        user_id: str,
        task_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Pull candidate experiences across all three scopes for the retriever."""
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    """
                    SELECT id, scope, workspace_id, user_id, task_type, tags,
                           lesson, failure_modes, successful_patterns,
                           outcome_label, weight, trust_score, conf_count,
                           last_confirmed_run, last_confirmed
                    FROM experience_records
                    WHERE archived = FALSE
                      AND weight > 0
                      AND (
                          (scope = 'project' AND workspace_id = $1::uuid)
                       OR (scope = 'user'    AND user_id = $2::uuid)
                       OR (scope = 'global')
                      )
                      AND ($3::text IS NULL OR task_type = $3)
                    ORDER BY trust_score DESC, last_confirmed DESC
                    LIMIT $4
                    """,
                    workspace_id or "00000000-0000-0000-0000-000000000000",
                    user_id or "00000000-0000-0000-0000-000000000000",
                    task_type,
                    limit,
                )
                result = []
                for r in rows:
                    d = dict(r)
                    for k in ("failure_modes", "successful_patterns"):
                        v = d.get(k)
                        if isinstance(v, str):
                            try:
                                d[k] = json.loads(v)
                            except (json.JSONDecodeError, TypeError):
                                d[k] = []
                    result.append(d)
                return result
            except Exception as exc:
                log.warning("db.exp_retrieve_error", error=str(exc))
                return []

    async def upsert_experience(self, record: dict[str, Any]) -> None:
        if not self._pool:
            return
        emb = record.get("lesson_emb")
        emb_literal = _pgvector_literal(emb) if emb else None
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO experience_records (
                        id, scope, workspace_id, user_id,
                        task_type, tags, lesson, lesson_emb,
                        failure_modes, successful_patterns, outcome_label,
                        weight, trust_score, conf_count,
                        last_confirmed_run, last_confirmed, archived
                    ) VALUES (
                        $1, $2, $3, $4,
                        $5, $6, $7, $8::vector,
                        $9::jsonb, $10::jsonb, $11,
                        $12, $13, $14,
                        $15, NOW(), $16
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        lesson = EXCLUDED.lesson,
                        weight = EXCLUDED.weight,
                        trust_score = EXCLUDED.trust_score,
                        conf_count = EXCLUDED.conf_count,
                        last_confirmed_run = EXCLUDED.last_confirmed_run,
                        last_confirmed = NOW(),
                        archived = EXCLUDED.archived
                    """,
                    record["id"],
                    record.get("scope", "project"),
                    record.get("workspace_id") or "00000000-0000-0000-0000-000000000000",
                    record.get("user_id") or "00000000-0000-0000-0000-000000000000",
                    record.get("task_type", "new_code"),
                    record.get("tags", []),
                    record.get("lesson", ""),
                    emb_literal,
                    json.dumps(record.get("failure_modes", [])),
                    json.dumps(record.get("successful_patterns", [])),
                    record.get("outcome_label", "success"),
                    int(record.get("weight", 2)),
                    float(record.get("trust_score", 1.0)),
                    int(record.get("conf_count", 1)),
                    int(record.get("last_confirmed_run", 0)),
                    bool(record.get("archived", False)),
                )
            except Exception as exc:
                log.warning("db.exp_upsert_error", error=str(exc))

    async def bulk_update_experience_stats(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        """Apply weight/trust/conf_count updates after retrieval/extraction."""
        if not self._pool or not records:
            return
        async with self._pool.acquire() as conn:
            for r in records:
                try:
                    await conn.execute(
                        """
                        UPDATE experience_records SET
                            weight = $2,
                            trust_score = $3,
                            conf_count = $4,
                            last_confirmed_run = $5,
                            last_confirmed = NOW(),
                            archived = $6
                        WHERE id = $1
                        """,
                        r["id"],
                        int(r.get("weight", 0)),
                        float(r.get("trust_score", 0.0)),
                        int(r.get("conf_count", 0)),
                        int(r.get("last_confirmed_run", 0)),
                        bool(r.get("archived", False)),
                    )
                except Exception as exc:
                    log.warning("db.exp_update_error", error=str(exc))

    # ────────────────────────────────────────────────────────────────────────
    # Bandit state (§5.3.12)
    # ────────────────────────────────────────────────────────────────────────

    async def load_bandit_state(self, workspace_id: str) -> dict[str, dict[str, Any]]:
        """Return {arm_id: {a_matrix, b_vector, n_observations}}."""
        if not self._pool:
            return {}
        async with self._pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    """
                    SELECT arm_id, d, a_matrix, b_vector, n_observations
                    FROM bandit_state
                    WHERE workspace_id = $1::uuid
                    """,
                    workspace_id or "00000000-0000-0000-0000-000000000000",
                )
                out: dict[str, dict[str, Any]] = {}
                for r in rows:
                    d = dict(r)
                    for k in ("a_matrix", "b_vector"):
                        v = d.get(k)
                        if isinstance(v, str):
                            try:
                                d[k] = json.loads(v)
                            except (json.JSONDecodeError, TypeError):
                                pass
                    out[d["arm_id"]] = d
                return out
            except Exception as exc:
                log.warning("db.bandit_load_error", error=str(exc))
                return {}

    async def save_bandit_arm(
        self,
        workspace_id: str,
        arm_id: str,
        d: int,
        a_matrix: list[list[float]],
        b_vector: list[float],
        n_observations: int,
    ) -> None:
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO bandit_state (
                        workspace_id, arm_id, d, a_matrix, b_vector, n_observations
                    ) VALUES (
                        $1::uuid, $2, $3, $4::jsonb, $5::jsonb, $6
                    )
                    ON CONFLICT (workspace_id, arm_id) DO UPDATE SET
                        d = EXCLUDED.d,
                        a_matrix = EXCLUDED.a_matrix,
                        b_vector = EXCLUDED.b_vector,
                        n_observations = EXCLUDED.n_observations,
                        updated_at = NOW()
                    """,
                    workspace_id or "00000000-0000-0000-0000-000000000000",
                    arm_id,
                    d,
                    json.dumps(a_matrix),
                    json.dumps(b_vector),
                    n_observations,
                )
            except Exception as exc:
                log.warning("db.bandit_save_error", error=str(exc))


db_client = DBClient()
