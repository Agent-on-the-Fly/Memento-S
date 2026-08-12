# SPDX-License-Identifier: Apache-2.0
"""add conversation_id to conversations

Revision ID: 001_add_conversation_id
Revises:
Create Date: 2026-04-20

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "001_add_conversation_id"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 检查列是否已存在（幂等：支持重试场景）
    from sqlalchemy import inspect

    inspector = inspect(op.get_bind())
    columns = [c["name"] for c in inspector.get_columns("conversations")]
    if "conversation_id" not in columns:
        op.add_column(
            "conversations",
            sa.Column("conversation_id", sa.String(36), nullable=False, server_default=""),
        )
        # 为现有数据填充空的 conversation_id（实际使用中应为每条记录生成唯一ID）
        # 这里用空字符串填充，查询 COUNT(DISTINCT conversation_id) 时会忽略
        op.execute("UPDATE conversations SET conversation_id = id WHERE conversation_id = ''")
        # 移除 server_default，改为 NOT NULL
        # SQLite 不支持 ALTER COLUMN DROP DEFAULT，需要用 batch 模式重建列
        with op.batch_alter_table("conversations") as batch_op:
            batch_op.alter_column("conversation_id", type_=sa.String(36), nullable=False)
    op.create_index("idx_conversation_conv_id", "conversations", ["conversation_id"], if_not_exists=True)


def downgrade() -> None:
    op.drop_index("idx_conversation_conv_id", table_name="conversations")
    op.drop_column("conversations", "conversation_id")