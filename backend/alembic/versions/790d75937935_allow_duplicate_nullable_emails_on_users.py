"""allow duplicate/nullable emails on users

Revision ID: 790d75937935
Revises: acbc6ba803e8
Create Date: 2025-10-05 10:18:44.384944

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '790d75937935'
down_revision: Union[str, Sequence[str], None] = 'acbc6ba803e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # 1) Thả unique index/constraint cũ trên email (tên của bạn là ix_users_email)
    try:
        op.drop_index('ix_users_email', table_name='users')
    except Exception:
        # nếu có unique constraint thay vì index:
        try:
            op.drop_constraint('uq_users_email', 'users', type_='unique')
        except Exception:
            pass

    # 2) Cho phép NULL trên users.email (nếu trước đây NOT NULL)
    with op.batch_alter_table('users') as batch_op:
        batch_op.alter_column('email', existing_type=sa.String(length=255), nullable=True)

    # 3) Tạo lại index thường (không unique)
    op.create_index('ix_users_email', 'users', ['email'], unique=False)


def downgrade():
    # quay lại: bỏ index thường, tạo unique lại (nếu muốn)
    op.drop_index('ix_users_email', table_name='users')
    with op.batch_alter_table('users') as batch_op:
        batch_op.alter_column('email', existing_type=sa.String(length=255), nullable=False)
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
