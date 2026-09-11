"""add benchmark tables

Revision ID: b3f8a1d94e70
Revises: d4c1a8e37b62
Create Date: 2026-09-07
"""

from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = 'b3f8a1d94e70'
down_revision: Union[str, None] = 'd4c1a8e37b62'
branch_labels = None
depends_on = None


def _index_exists(inspector, index_name, table_name):
    """Check if an index already exists on the given table (works for both SQLite and PostgreSQL)."""
    indexes = inspector.get_indexes(table_name)
    return any(idx['name'] == index_name for idx in indexes)


def upgrade():
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    if 'benchmark_config' not in tables:
        op.create_table(
            'benchmark_config',
            sa.Column('config_id', sa.Text(), primary_key=True),
            sa.Column('alias', sa.Text(), nullable=False),
            sa.Column('config_text', sa.Text(), nullable=False),
            sa.Column('arch', sa.Text(), nullable=True),
            sa.Column('ngl', sa.Integer(), nullable=True),
            sa.Column('ctx', sa.Integer(), nullable=True),
            sa.Column('parallel', sa.Integer(), nullable=True),
            sa.Column('threads', sa.Integer(), nullable=True),
            sa.Column('moe', sa.Integer(), nullable=True),
            sa.Column('override_tensors', sa.Text(), nullable=True),
            sa.Column('speculative', sa.Text(), nullable=True),
            sa.Column('spec_draft_n_max', sa.Integer(), nullable=True),
            sa.Column('cache_k', sa.Text(), nullable=True),
            sa.Column('cache_v', sa.Text(), nullable=True),
            sa.Column('flash_attn', sa.Text(), nullable=True),
            sa.Column('batch', sa.Integer(), nullable=True),
            sa.Column('ubatch', sa.Integer(), nullable=True),
            sa.Column('reasoning_effort', sa.Text(), nullable=True),
            sa.Column('samplers', sa.Text(), nullable=True),
            sa.Column('first_seen', sa.BigInteger(), nullable=False),
        )

    if 'benchmark_run' not in tables:
        op.create_table(
            'benchmark_run',
            sa.Column('run_id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column('config_id', sa.Text(), sa.ForeignKey('benchmark_config.config_id'), nullable=False),
            sa.Column('model', sa.Text(), nullable=False),
            sa.Column('quant', sa.Text(), nullable=False),
            sa.Column('build', sa.Text(), nullable=False),
            sa.Column('port', sa.Integer(), nullable=False),
            sa.Column('pid', sa.Integer(), nullable=True),
            sa.Column('started_at', sa.BigInteger(), nullable=False),
            sa.Column('ended_at', sa.BigInteger(), nullable=True),
            sa.Column('ended_reason', sa.Text(), nullable=True),
        )

    inspector.clear_cache()
    if 'benchmark_run' in inspector.get_table_names():
        if not _index_exists(inspector, 'ix_benchmark_run_config', 'benchmark_run'):
            op.create_index('ix_benchmark_run_config', 'benchmark_run', ['config_id', 'started_at'])
        if not _index_exists(inspector, 'ix_benchmark_run_active', 'benchmark_run'):
            op.create_index(
                'ix_benchmark_run_active', 'benchmark_run', ['port'], postgresql_where=sa.text('ended_at IS NULL')
            )

    if 'benchmark_run_load_info' not in tables:
        op.create_table(
            'benchmark_run_load_info',
            sa.Column(
                'run_id',
                sa.Integer(),
                sa.ForeignKey('benchmark_run.run_id', ondelete='CASCADE'),
                primary_key=True,
            ),
            sa.Column('n_layer', sa.Integer(), nullable=True),
            sa.Column('n_layer_all', sa.Integer(), nullable=True),
            sa.Column('layers_gpu', sa.Integer(), nullable=True),
            sa.Column('layers_total', sa.Integer(), nullable=True),
            sa.Column('layers_derived', sa.Integer(), nullable=True),
            sa.Column('n_slots', sa.Integer(), nullable=True),
            sa.Column('n_ctx_slot', sa.Integer(), nullable=True),
            sa.Column('kv_unified', sa.Text(), nullable=True),
            sa.Column('fused_gdn', sa.Text(), nullable=True),
            sa.Column('mtp_head', sa.Text(), nullable=True),
            sa.Column('buffers', sa.JSON(), nullable=True),
            sa.Column('cpu_buffer_mib', sa.Float(), nullable=True),
            sa.Column('gpu_buffer_mib', sa.Float(), nullable=True),
            sa.Column('unused_tensors', sa.Integer(), nullable=True),
            sa.Column('unused_prefixes', sa.JSON(), nullable=True),
            sa.Column('warnings', sa.JSON(), nullable=True),
            sa.Column('deprecated', sa.JSON(), nullable=True),
        )

    if 'benchmark_gpu_sample' not in tables:
        op.create_table(
            'benchmark_gpu_sample',
            sa.Column('sample_id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                'run_id', sa.Integer(), sa.ForeignKey('benchmark_run.run_id', ondelete='CASCADE'), nullable=False
            ),
            sa.Column('at', sa.BigInteger(), nullable=False),
            sa.Column('temp_c', sa.Integer(), nullable=True),
            sa.Column('util_pct', sa.Integer(), nullable=True),
            sa.Column('mem_used_mib', sa.Integer(), nullable=True),
            sa.Column('mem_total_mib', sa.Integer(), nullable=True),
            sa.Column('power_w', sa.Float(), nullable=True),
            sa.Column('sm_mhz', sa.Integer(), nullable=True),
            sa.Column('throttle', sa.Integer(), nullable=True),
        )

    inspector.clear_cache()
    if 'benchmark_gpu_sample' in inspector.get_table_names():
        if not _index_exists(inspector, 'ix_benchmark_gpu_sample_run', 'benchmark_gpu_sample'):
            op.create_index('ix_benchmark_gpu_sample_run', 'benchmark_gpu_sample', ['run_id', 'at'])

    if 'benchmark_metrics_scrape' not in tables:
        op.create_table(
            'benchmark_metrics_scrape',
            sa.Column(
                'run_id', sa.Integer(), sa.ForeignKey('benchmark_run.run_id', ondelete='CASCADE'), primary_key=True
            ),
            sa.Column('at', sa.BigInteger(), primary_key=True),
            sa.Column('counter', sa.Text(), primary_key=True),
            sa.Column('value', sa.Float(), nullable=False),
        )

    if 'benchmark_request' not in tables:
        op.create_table(
            'benchmark_request',
            sa.Column('request_id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                'run_id', sa.Integer(), sa.ForeignKey('benchmark_run.run_id', ondelete='CASCADE'), nullable=True
            ),
            sa.Column('at', sa.BigInteger(), nullable=False),
            sa.Column('model', sa.Text(), nullable=True),
            sa.Column('label', sa.Text(), nullable=True),
            sa.Column('wall_ms', sa.Float(), nullable=True),
            sa.Column('params', sa.JSON(), nullable=True),
            sa.Column('cache_n', sa.Integer(), nullable=True),
            sa.Column('prompt_n', sa.Integer(), nullable=True),
            sa.Column('prompt_ms', sa.Float(), nullable=True),
            sa.Column('prompt_per_token_ms', sa.Float(), nullable=True),
            sa.Column('prompt_per_second', sa.Float(), nullable=True),
            sa.Column('predicted_n', sa.Integer(), nullable=True),
            sa.Column('predicted_ms', sa.Float(), nullable=True),
            sa.Column('predicted_per_token_ms', sa.Float(), nullable=True),
            sa.Column('predicted_per_second', sa.Float(), nullable=True),
            sa.Column('draft_n', sa.Integer(), nullable=True),
            sa.Column('draft_n_accepted', sa.Integer(), nullable=True),
            sa.Column('timings', sa.JSON(), nullable=True),
        )

    inspector.clear_cache()
    if 'benchmark_request' in inspector.get_table_names():
        if not _index_exists(inspector, 'ix_benchmark_request_run', 'benchmark_request'):
            op.create_index('ix_benchmark_request_run', 'benchmark_request', ['run_id', 'at'])

    if 'benchmark_result' not in tables:
        op.create_table(
            'benchmark_result',
            sa.Column('result_id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column('suite_run_id', sa.Text(), nullable=False),
            sa.Column(
                'run_id', sa.Integer(), sa.ForeignKey('benchmark_run.run_id', ondelete='SET NULL'), nullable=True
            ),
            sa.Column(
                'request_id',
                sa.Integer(),
                sa.ForeignKey('benchmark_request.request_id', ondelete='SET NULL'),
                nullable=True,
            ),
            sa.Column('config_id', sa.Text(), sa.ForeignKey('benchmark_config.config_id'), nullable=True),
            sa.Column('at', sa.BigInteger(), nullable=False),
            sa.Column('model', sa.Text(), nullable=False),
            sa.Column('profile', sa.Text(), nullable=True),
            sa.Column('benchmark', sa.Text(), nullable=False),
            sa.Column('item_id', sa.Text(), nullable=False),
            sa.Column('dataset_revision', sa.Text(), nullable=False),
            sa.Column('tier', sa.Text(), nullable=False),
            sa.Column('seed', sa.Integer(), nullable=False),
            sa.Column('outcome', sa.Text(), nullable=False),
            sa.Column('reason', sa.Text(), nullable=False, server_default=''),
            sa.Column('reasoning_chars', sa.Integer(), nullable=True),
            sa.Column('wall_ms', sa.Float(), nullable=True),
            sa.Column('params', sa.JSON(), nullable=True),
            sa.Column('timings', sa.JSON(), nullable=True),
            sa.Column('system_name', sa.Text(), nullable=True),
            sa.Column('system_sha', sa.Text(), nullable=True),
            sa.Column('adapter_sha', sa.Text(), nullable=True),
            sa.UniqueConstraint('suite_run_id', 'benchmark', 'item_id', name='uq_benchmark_result_item'),
            sa.CheckConstraint(
                "outcome IN ('pass','fail_assert','fail_error','fail_timeout','no_code','skipped')",
                name='ck_benchmark_result_outcome',
            ),
        )

    inspector.clear_cache()
    if 'benchmark_result' in inspector.get_table_names():
        if not _index_exists(inspector, 'ix_benchmark_result_group', 'benchmark_result'):
            op.create_index(
                'ix_benchmark_result_group', 'benchmark_result', ['model', 'config_id', 'tier', 'benchmark']
            )

    if 'benchmark_answer' not in tables:
        op.create_table(
            'benchmark_answer',
            sa.Column(
                'result_id',
                sa.Integer(),
                sa.ForeignKey('benchmark_result.result_id', ondelete='CASCADE'),
                primary_key=True,
            ),
            sa.Column('prompt', sa.Text(), nullable=False),
            sa.Column('content', sa.Text(), nullable=False),
            sa.Column('reasoning', sa.Text(), nullable=False),
        )

    if 'benchmark_suite_exclusion' not in tables:
        op.create_table(
            'benchmark_suite_exclusion',
            sa.Column('benchmark', sa.Text(), primary_key=True),
            sa.Column('item_id', sa.Text(), primary_key=True),
            sa.Column('dataset_revision', sa.Text(), primary_key=True),
            sa.Column('kind', sa.Text(), nullable=False),
            sa.Column('reason', sa.Text(), nullable=False),
            sa.Column('recorded_at', sa.BigInteger(), nullable=False),
        )

    if 'benchmark_schema_note' not in tables:
        op.create_table(
            'benchmark_schema_note',
            sa.Column('note_id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column('noted_on', sa.Text(), nullable=False),
            sa.Column('note', sa.Text(), nullable=False, unique=True),
        )

    if 'benchmark_tune_sweep' not in tables:
        op.create_table(
            'benchmark_tune_sweep',
            sa.Column('sweep_id', sa.Text(), primary_key=True),
            sa.Column('started_at', sa.BigInteger(), nullable=False),
            sa.Column('ended_at', sa.BigInteger(), nullable=True),
            sa.Column('ended_reason', sa.Text(), nullable=True),
            sa.Column('pid', sa.Integer(), nullable=True),
            sa.Column('profile', sa.Text(), nullable=False),
            sa.Column('tier', sa.Text(), nullable=False),
            sa.Column('benchmark', sa.Text(), nullable=True),
            sa.Column('system_name', sa.Text(), nullable=True),
            sa.Column('system_sha', sa.Text(), nullable=True),
            sa.Column('grid_path', sa.Text(), nullable=False),
            sa.Column('grid_sha', sa.Text(), nullable=False),
            sa.Column('item_order_sha', sa.Text(), nullable=False),
            sa.Column('item_count', sa.Integer(), nullable=False),
            sa.Column('budget_mode', sa.Text(), nullable=False),
            sa.Column('budget_seconds', sa.Integer(), nullable=True),
            sa.Column('budget_visits', sa.Integer(), nullable=True),
            sa.Column('eta', sa.Integer(), nullable=False),
            sa.Column('round_items', sa.Integer(), nullable=False),
            sa.Column('candidates', sa.Integer(), nullable=False),
            sa.Column('stages', sa.Text(), nullable=False, server_default='explore,refine'),
            sa.Column('objective', sa.Text(), nullable=False),
            sa.Column('alpha', sa.Float(), nullable=False),
            sa.Column('on_drift', sa.Text(), nullable=False),
            sa.Column('seed', sa.Integer(), nullable=False),
            sa.Column('baseline_config_id', sa.Text(), sa.ForeignKey('benchmark_config.config_id'), nullable=True),
            sa.Column('winner_candidate', sa.Text(), nullable=True),
            sa.Column('verdict', sa.Text(), nullable=True),
            sa.Column('verdict_reason', sa.Text(), nullable=False, server_default=''),
            sa.CheckConstraint(
                "budget_mode IN ('interactive','overnight','multiday','trials')",
                name='ck_benchmark_tune_sweep_budget_mode',
            ),
            sa.CheckConstraint(
                "verdict IS NULL OR verdict IN ('adopted','rejected','indeterminate','incomplete')",
                name='ck_benchmark_tune_sweep_verdict',
            ),
        )

    if 'benchmark_tune_candidate' not in tables:
        op.create_table(
            'benchmark_tune_candidate',
            sa.Column(
                'sweep_id',
                sa.Text(),
                sa.ForeignKey('benchmark_tune_sweep.sweep_id', ondelete='CASCADE'),
                primary_key=True,
            ),
            sa.Column('candidate_sha', sa.Text(), primary_key=True),
            sa.Column('stage', sa.Text(), nullable=False),
            sa.Column('overrides', sa.JSON(), nullable=False),
            sa.Column('is_baseline', sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column('config_id', sa.Text(), sa.ForeignKey('benchmark_config.config_id'), nullable=True),
            sa.Column('suite_run_id', sa.Text(), nullable=False),
            sa.Column('status', sa.Text(), nullable=False),
            sa.Column('status_reason', sa.Text(), nullable=False, server_default=''),
            sa.Column('score', sa.Float(), nullable=True),
            sa.Column('eliminated_round', sa.Integer(), nullable=True),
            sa.Column('created_at', sa.BigInteger(), nullable=False),
            sa.CheckConstraint(
                "status IN ('pending','active','eliminated','infeasible','marginal','winner','rejected')",
                name='ck_benchmark_tune_candidate_status',
            ),
        )

    if 'benchmark_tune_round' not in tables:
        op.create_table(
            'benchmark_tune_round',
            sa.Column(
                'sweep_id',
                sa.Text(),
                sa.ForeignKey('benchmark_tune_sweep.sweep_id', ondelete='CASCADE'),
                primary_key=True,
            ),
            sa.Column('round', sa.Integer(), primary_key=True),
            sa.Column('stage', sa.Text(), nullable=False),
            sa.Column('started_at', sa.BigInteger(), nullable=False),
            sa.Column('ended_at', sa.BigInteger(), nullable=True),
            sa.Column('item_from', sa.Integer(), nullable=False),
            sa.Column('item_to', sa.Integer(), nullable=False),
            sa.Column('survivors', sa.Integer(), nullable=False),
            sa.Column('baseline_gen_tps', sa.Float(), nullable=True),
            sa.Column('baseline_regime', sa.Text(), nullable=True),
            sa.Column('drift_ratio', sa.Float(), nullable=True),
            sa.Column('decision', sa.Text(), nullable=False, server_default=''),
            sa.Column('notes', sa.Text(), nullable=False, server_default=''),
        )

    if 'benchmark_tune_visit' not in tables:
        op.create_table(
            'benchmark_tune_visit',
            sa.Column('visit_id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                'sweep_id',
                sa.Text(),
                sa.ForeignKey('benchmark_tune_sweep.sweep_id', ondelete='CASCADE'),
                nullable=False,
            ),
            sa.Column('candidate_sha', sa.Text(), nullable=False),
            sa.Column('round', sa.Integer(), nullable=False),
            sa.Column('attempt', sa.Integer(), nullable=False, server_default='1'),
            sa.Column('run_id', sa.Integer(), sa.ForeignKey('benchmark_run.run_id', ondelete='SET NULL'), nullable=True),
            sa.Column('config_id', sa.Text(), nullable=True),
            sa.Column('started_at', sa.BigInteger(), nullable=False),
            sa.Column('ended_at', sa.BigInteger(), nullable=True),
            sa.Column('server_pgid', sa.Integer(), nullable=True),
            sa.Column('load_ms', sa.Float(), nullable=True),
            sa.Column('item_from', sa.Integer(), nullable=False),
            sa.Column('item_to', sa.Integer(), nullable=False),
            sa.Column('items_done', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('since_pause_seconds', sa.Float(), nullable=True),
            sa.Column('counts_toward_round', sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column('status', sa.Text(), nullable=False),
            sa.Column('reason', sa.Text(), nullable=False, server_default=''),
            sa.UniqueConstraint(
                'sweep_id', 'candidate_sha', 'round', 'attempt', name='uq_benchmark_tune_visit_slot'
            ),
            sa.CheckConstraint(
                "status IN ('running','done','infeasible','aborted')", name='ck_benchmark_tune_visit_status'
            ),
        )

    inspector.clear_cache()
    if 'benchmark_tune_visit' in inspector.get_table_names():
        if not _index_exists(inspector, 'ix_benchmark_tune_visit_sweep', 'benchmark_tune_visit'):
            op.create_index('ix_benchmark_tune_visit_sweep', 'benchmark_tune_visit', ['sweep_id', 'round'])

    if 'benchmark_tune_pause' not in tables:
        op.create_table(
            'benchmark_tune_pause',
            sa.Column('pause_id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                'sweep_id',
                sa.Text(),
                sa.ForeignKey('benchmark_tune_sweep.sweep_id', ondelete='CASCADE'),
                nullable=False,
            ),
            sa.Column('round', sa.Integer(), nullable=True),
            sa.Column(
                'visit_id',
                sa.Integer(),
                sa.ForeignKey('benchmark_tune_visit.visit_id', ondelete='SET NULL'),
                nullable=True,
            ),
            sa.Column('started_at', sa.BigInteger(), nullable=False),
            sa.Column('ended_at', sa.BigInteger(), nullable=True),
            sa.Column('trigger_kind', sa.Text(), nullable=False),
            sa.Column('drift_ratio', sa.Float(), nullable=True),
            sa.Column('attempt', sa.Integer(), nullable=False, server_default='1'),
            sa.Column('throttle_before', sa.Text(), nullable=True),
            sa.Column('throttle_after', sa.Text(), nullable=True),
            sa.Column('temp_before', sa.Float(), nullable=True),
            sa.Column('temp_after', sa.Float(), nullable=True),
            sa.Column('power_before', sa.Float(), nullable=True),
            sa.Column('power_after', sa.Float(), nullable=True),
            sa.Column('probe_tps', sa.Float(), nullable=True),
            sa.Column('resolution', sa.Text(), nullable=True),
            sa.CheckConstraint("trigger_kind IN ('drift','regime','cliff')", name='ck_benchmark_tune_pause_trigger'),
            sa.CheckConstraint(
                "resolution IS NULL OR resolution IN ('recovered','timeout','abandoned')",
                name='ck_benchmark_tune_pause_resolution',
            ),
        )

    inspector.clear_cache()
    if 'benchmark_tune_pause' in inspector.get_table_names():
        if not _index_exists(inspector, 'ix_benchmark_tune_pause_sweep', 'benchmark_tune_pause'):
            op.create_index('ix_benchmark_tune_pause_sweep', 'benchmark_tune_pause', ['sweep_id', 'started_at'])


def downgrade():
    op.drop_index('ix_benchmark_tune_pause_sweep', table_name='benchmark_tune_pause')
    op.drop_table('benchmark_tune_pause')
    op.drop_index('ix_benchmark_tune_visit_sweep', table_name='benchmark_tune_visit')
    op.drop_table('benchmark_tune_visit')
    op.drop_table('benchmark_tune_round')
    op.drop_table('benchmark_tune_candidate')
    op.drop_table('benchmark_tune_sweep')
    op.drop_table('benchmark_schema_note')
    op.drop_table('benchmark_suite_exclusion')
    op.drop_table('benchmark_answer')
    op.drop_index('ix_benchmark_result_group', table_name='benchmark_result')
    op.drop_table('benchmark_result')
    op.drop_index('ix_benchmark_request_run', table_name='benchmark_request')
    op.drop_table('benchmark_request')
    op.drop_table('benchmark_metrics_scrape')
    op.drop_index('ix_benchmark_gpu_sample_run', table_name='benchmark_gpu_sample')
    op.drop_table('benchmark_gpu_sample')
    op.drop_table('benchmark_run_load_info')
    op.drop_index('ix_benchmark_run_active', table_name='benchmark_run')
    op.drop_index('ix_benchmark_run_config', table_name='benchmark_run')
    op.drop_table('benchmark_run')
    op.drop_table('benchmark_config')
