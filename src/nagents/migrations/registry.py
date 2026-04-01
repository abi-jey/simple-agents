"""Migration registry - centralized storage of all migrations."""

from .base import Migration


class MigrationRegistry:
    """Registry for managing migrations across different database types.

    This class provides a way to register and retrieve migrations for
    different database types (sessions, batch, etc.).

    Example:
        registry = MigrationRegistry()
        registry.register("sessions", [
            Migration(1, "Initial schema", "..."),
            Migration(2, "Add compaction", "..."),
        ])
        migrations = registry.get("sessions")
    """

    def __init__(self) -> None:
        self._migrations: dict[str, list[Migration]] = {}

    def register(self, db_type: str, migrations: list[Migration]) -> None:
        """Register migrations for a database type.

        Args:
            db_type: Database type identifier (e.g., "sessions", "batch")
            migrations: List of migrations in version order
        """
        self._migrations[db_type] = sorted(migrations, key=lambda m: m.version)

    def get(self, db_type: str) -> list[Migration]:
        """Get migrations for a specific database type.

        Args:
            db_type: Database type identifier

        Returns:
            List of migrations for the database type
        """
        return self._migrations.get(db_type, [])

    def get_all(self) -> dict[str, list[Migration]]:
        """Get all registered migrations.

        Returns:
            Dict mapping db_type to list of migrations
        """
        return self._migrations.copy()

    def get_latest_version(self, db_type: str) -> int:
        """Get the latest migration version for a database type.

        Args:
            db_type: Database type identifier

        Returns:
            Latest version number, or 0 if no migrations
        """
        migrations = self.get(db_type)
        return max((m.version for m in migrations), default=0)


__all__ = ["Migration", "MigrationRegistry"]
