"""Session database migrations.

This module defines all migrations for the sessions database, from initial
schema through all version updates.
"""

from .base import Migration

# Version 1: Initial schema (v2_sessions and v2_messages tables)
# This represents the original schema that was created on first initialize()
MIGRATION_001_INITIAL = Migration(
    version=1,
    description="Initial schema with v2_sessions and v2_messages tables",
    up_sql="""
        CREATE TABLE IF NOT EXISTS v2_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS v2_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_calls TEXT,
            tool_call_id TEXT,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES v2_sessions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_v2_messages_session
        ON v2_messages(session_id);

        CREATE INDEX IF NOT EXISTS idx_v2_sessions_user
        ON v2_sessions(user_id);
    """,
    down_sql="""
        DROP INDEX IF EXISTS idx_v2_sessions_user;
        DROP INDEX IF EXISTS idx_v2_messages_session;
        DROP TABLE IF EXISTS v2_messages;
        DROP TABLE IF EXISTS v2_sessions;
    """,
)

# Version 2: Add compacted_at_message_id column for compaction boundary
# This allows tracking where compaction summary was stored
MIGRATION_002_COMPACTION = Migration(
    version=2,
    description="Add compacted_at_message_id column for compaction boundary",
    up_sql="""
        ALTER TABLE v2_sessions ADD COLUMN compacted_at_message_id INTEGER;
    """,
    down_sql="""
        -- SQLite doesn't support DROP COLUMN directly, so we recreate the table
        CREATE TABLE v2_sessions_backup AS
        SELECT id, user_id, created_at, updated_at FROM v2_sessions;

        DROP TABLE v2_sessions;

        CREATE TABLE v2_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        INSERT INTO v2_sessions SELECT * FROM v2_sessions_backup;

        DROP TABLE v2_sessions_backup;

        CREATE INDEX IF NOT EXISTS idx_v2_sessions_user ON v2_sessions(user_id);
    """,
)


migrations = [
    MIGRATION_001_INITIAL,
    MIGRATION_002_COMPACTION,
]

__all__ = ["MIGRATION_001_INITIAL", "MIGRATION_002_COMPACTION", "migrations"]
