# app/database/__init__.py
# Marks the `database` directory as a Python package.
# All database connectivity and session management will live here in future phases.
# Planned contents:
#   - connection.py   (async SQLAlchemy engine + session factory)
#   - base.py         (declarative Base for ORM models)
#   - migrations/     (Alembic migration scripts)
# The session dependency will be injected into FastAPI route handlers via Depends().
