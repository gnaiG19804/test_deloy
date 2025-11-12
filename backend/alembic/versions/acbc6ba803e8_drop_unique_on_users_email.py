"""drop unique on users.email

Revision ID: acbc6ba803e8
Revises: init_users_auth
Create Date: 2025-10-05 10:06:07.999125

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'acbc6ba803e8'
down_revision: Union[str, Sequence[str], None] = 'init_users_auth'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # tên index/constraint có thể là "ix_users_email" hoặc "uq_users_email" tùy bạn tạo
    # Kiểm tra thực tế bằng \d users (psql)
    op.drop_index('ix_users_email', table_name='users')  # nếu index unique tên vậy
    # Nếu unique constraint riêng:
    # op.drop_constraint('uq_users_email', 'users', type_='unique')
    # Tạo lại index thường (không unique)
    op.create_index('ix_users_email', 'users', ['email'], unique=False)


def downgrade():
    op.drop_index('ix_users_email', table_name='users')
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
