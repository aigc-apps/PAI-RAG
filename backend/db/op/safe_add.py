from alembic import op
import sqlalchemy as sa

def safe_add_column(table_name, column: sa.Column):
    # 先检查列是否存在（Sqlite 不支持add column if not exists）
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    # Get column info
    columns = inspector.get_columns(table_name)

    # Example: just get column names as a list
    column_names = [col['name'] for col in columns]

    if column.name not in column_names:
        op.add_column(
            table_name,
            column,
        )
