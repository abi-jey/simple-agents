"""Migration base class and types."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Migration:
    """Represents a single database migration.

    Attributes:
        version: Migration version number (must be sequential)
        description: Human-readable description
        up_sql: SQL to apply the migration
        down_sql: SQL to rollback the migration (optional)
    """

    version: int
    description: str
    up_sql: str
    down_sql: str | None = None

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("Migration version must be >= 1")

    def __lt__(self, other: "Migration") -> bool:
        return self.version < other.version

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Migration):
            return NotImplemented
        return self.version == other.version

    def __hash__(self) -> int:
        return hash(self.version)
