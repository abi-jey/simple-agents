"""Database migration system for nagents.

This module provides a migration system for managing database schema changes
with version tracking and rollback support.

Key features:
- Automatic migration on initialize()
- Version tracking via schema_migrations table
- Rollback support with stored rollback SQL
- Migration history tracking
- Database-specific migrations (sessions, batch)

Example:
    from nagents.session import SessionManager

    # Automatic migration on initialize
    sm = SessionManager(Path("sessions.db"))
    await sm.initialize()

    # Check schema version
    version = await sm.get_schema_version()

    # Get migration history
    history = await sm.get_migration_history()

    # Rollback last migration
    await sm.rollback_migration()
"""

from .base import Migration
from .manager import MigrationManager
from .registry import MigrationRegistry

__all__ = [
    "Migration",
    "MigrationManager",
    "MigrationRegistry",
]
