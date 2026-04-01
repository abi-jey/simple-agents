"""Batch database migrations.

This module defines all migrations for the batch jobs database.
"""

from .base import Migration

# Version 1: Initial schema for batch jobs
MIGRATION_001_INITIAL = Migration(
    version=1,
    description="Initial schema for batch jobs and requests",
    up_sql="""
        -- Core batch job tracking
        CREATE TABLE IF NOT EXISTS v2_batch_jobs (
            id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            status TEXT NOT NULL,
            model TEXT NOT NULL,

            -- Request counts
            total_requests INTEGER DEFAULT 0,
            succeeded INTEGER DEFAULT 0,
            failed INTEGER DEFAULT 0,
            cancelled INTEGER DEFAULT 0,
            expired INTEGER DEFAULT 0,

            -- Timestamps (ISO format)
            created_at TEXT,
            expires_at TEXT,
            completed_at TEXT,

            -- Provider-specific IDs
            input_file_id TEXT,
            output_file_id TEXT,
            error_file_id TEXT,
            results_url TEXT,

            -- Processing state
            processed INTEGER DEFAULT 0,
            processed_at TEXT,

            -- Callback for result handling
            callback_name TEXT,
            callback_context TEXT,

            -- API connection info (for resuming)
            base_url TEXT,

            -- Metadata
            metadata TEXT,

            -- Tracking
            db_created_at TEXT DEFAULT (datetime('now')),
            db_updated_at TEXT DEFAULT (datetime('now'))
        );

        -- Store original requests for retry capability
        CREATE TABLE IF NOT EXISTS v2_batch_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            custom_id TEXT NOT NULL,
            messages TEXT NOT NULL,
            model TEXT,
            tools TEXT,
            config TEXT,
            system_prompt TEXT,

            -- Result tracking
            result_type TEXT,
            result_content TEXT,
            result_error TEXT,

            FOREIGN KEY (batch_id) REFERENCES v2_batch_jobs(id),
            UNIQUE (batch_id, custom_id)
        );

        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_v2_batch_jobs_status
        ON v2_batch_jobs(status);

        CREATE INDEX IF NOT EXISTS idx_v2_batch_jobs_processed
        ON v2_batch_jobs(processed);

        CREATE INDEX IF NOT EXISTS idx_v2_batch_jobs_provider
        ON v2_batch_jobs(provider);

        CREATE INDEX IF NOT EXISTS idx_v2_batch_requests_batch
        ON v2_batch_requests(batch_id);

        CREATE INDEX IF NOT EXISTS idx_v2_batch_requests_custom_id
        ON v2_batch_requests(batch_id, custom_id);
    """,
    down_sql="""
        DROP INDEX IF EXISTS idx_v2_batch_requests_custom_id;
        DROP INDEX IF EXISTS idx_v2_batch_requests_batch;
        DROP INDEX IF EXISTS idx_v2_batch_jobs_provider;
        DROP INDEX IF EXISTS idx_v2_batch_jobs_processed;
        DROP INDEX IF EXISTS idx_v2_batch_jobs_status;
        DROP TABLE IF EXISTS v2_batch_requests;
        DROP TABLE IF EXISTS v2_batch_jobs;
    """,
)


migrations = [
    MIGRATION_001_INITIAL,
]

__all__ = ["MIGRATION_001_INITIAL", "migrations"]
