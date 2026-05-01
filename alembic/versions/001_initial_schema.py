"""initial schema

Revision ID: 001
Revises: None
Create Date: 2026-05-01

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'action_log',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text('NULL')), # UUID handled by SQLAlchemy
        sa.Column('task_id', sa.UUID(), nullable=False),
        sa.Column('step_id', sa.String(length=64)),
        sa.Column('action_type', sa.String(length=32)),
        sa.Column('node_id', sa.String(length=128)),
        sa.Column('app_package', sa.String(length=128)),
        sa.Column('status', sa.String(length=16)),
        sa.Column('hitl_required', sa.Boolean(), server_default=sa.text('0')),
        sa.Column('hitl_approved', sa.Boolean()),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now())
    )
    op.create_index('idx_action_log_task', 'action_log', ['task_id'])

    op.create_table(
        'sessions',
        sa.Column('session_id', sa.UUID(), primary_key=True),
        sa.Column('connected_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('disconnected_at', sa.DateTime(timezone=True)),
        sa.Column('token_balance', sa.Integer(), server_default=sa.text('100'))
    )

def downgrade():
    op.drop_table('sessions')
    op.drop_table('action_log')
