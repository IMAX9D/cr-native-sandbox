"""SQLite WAL lease queue for asynchronous work stealing and resume."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ClaimedTask:
    battle_tag: str
    source_path: str
    source_sha256: str
    attempts: int
    payload: dict[str, Any]


class TickStoreWorkQueue:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.path, timeout=30.0, isolation_level=None
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                battle_tag TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending','leased','done','failed')),
                lease_owner TEXT,
                lease_until REAL,
                attempts INTEGER NOT NULL DEFAULT 0,
                output_shard TEXT,
                frame_offset INTEGER,
                frame_size INTEGER,
                episode_sha256 TEXT,
                last_error TEXT,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS tasks_status_lease
                ON tasks(status, lease_until, attempts, battle_tag);
            """
        )

    def add_tasks(self, tasks: Iterable[Mapping[str, Any]]) -> int:
        now = time.time()
        inserted = 0
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            for task in tasks:
                tag = str(task["battle_tag"])
                source_path = str(task["source_path"])
                source_sha = str(task["source_sha256"])
                payload = json.dumps(
                    dict(task.get("payload") or {}),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                row = self.connection.execute(
                    "SELECT source_path, source_sha256, payload_json FROM tasks WHERE battle_tag=?",
                    (tag,),
                ).fetchone()
                if row is not None:
                    if (
                        row["source_path"] != source_path
                        or row["source_sha256"] != source_sha
                        or row["payload_json"] != payload
                    ):
                        raise RuntimeError(f"task identity changed for {tag}")
                    continue
                self.connection.execute(
                    """
                    INSERT INTO tasks(
                        battle_tag, source_path, source_sha256, payload_json, updated_at
                    ) VALUES(?,?,?,?,?)
                    """,
                    (tag, source_path, source_sha, payload, now),
                )
                inserted += 1
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return inserted

    def claim(
        self,
        worker_id: str,
        *,
        limit: int = 1,
        lease_seconds: float = 300.0,
        maximum_attempts: int = 5,
    ) -> list[ClaimedTask]:
        if limit <= 0 or lease_seconds <= 0:
            raise ValueError("claim limit and lease must be positive")
        now = time.time()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            # Expired work is globally stealable. Attempts are incremented only
            # when a worker actually acquires a lease.
            self.connection.execute(
                """
                UPDATE tasks SET status='pending', lease_owner=NULL,
                    lease_until=NULL, updated_at=?
                WHERE status='leased' AND lease_until < ? AND attempts < ?
                """,
                (now, now, maximum_attempts),
            )
            self.connection.execute(
                """
                UPDATE tasks SET status='failed', lease_owner=NULL,
                    lease_until=NULL, last_error='lease attempts exhausted', updated_at=?
                WHERE status='leased' AND lease_until < ? AND attempts >= ?
                """,
                (now, now, maximum_attempts),
            )
            rows = self.connection.execute(
                """
                SELECT battle_tag FROM tasks
                WHERE status='pending' AND attempts < ?
                ORDER BY attempts, battle_tag LIMIT ?
                """,
                (maximum_attempts, limit),
            ).fetchall()
            tags = [str(row["battle_tag"]) for row in rows]
            for tag in tags:
                self.connection.execute(
                    """
                    UPDATE tasks SET status='leased', lease_owner=?, lease_until=?,
                        attempts=attempts+1, updated_at=?
                    WHERE battle_tag=? AND status='pending'
                    """,
                    (worker_id, now + lease_seconds, now, tag),
                )
            result = []
            for tag in tags:
                row = self.connection.execute(
                    "SELECT * FROM tasks WHERE battle_tag=?", (tag,)
                ).fetchone()
                result.append(
                    ClaimedTask(
                        battle_tag=tag,
                        source_path=str(row["source_path"]),
                        source_sha256=str(row["source_sha256"]),
                        attempts=int(row["attempts"]),
                        payload=json.loads(row["payload_json"]),
                    )
                )
            self.connection.execute("COMMIT")
            return result
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def heartbeat(
        self, worker_id: str, battle_tags: Iterable[str], *, lease_seconds: float = 300.0
    ) -> int:
        tags = list(battle_tags)
        if not tags:
            return 0
        now = time.time()
        updated = 0
        with self.connection:
            for tag in tags:
                cursor = self.connection.execute(
                    """
                    UPDATE tasks SET lease_until=?, updated_at=?
                    WHERE battle_tag=? AND status='leased' AND lease_owner=?
                    """,
                    (now + lease_seconds, now, tag, worker_id),
                )
                updated += cursor.rowcount
        return updated

    def complete(
        self,
        worker_id: str,
        battle_tag: str,
        *,
        output_shard: str,
        frame_offset: int,
        frame_size: int,
        episode_sha256: str,
    ) -> None:
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE tasks SET status='done', lease_owner=NULL, lease_until=NULL,
                    output_shard=?, frame_offset=?, frame_size=?, episode_sha256=?,
                    updated_at=?
                WHERE battle_tag=? AND status='leased' AND lease_owner=?
                """,
                (
                    output_shard,
                    frame_offset,
                    frame_size,
                    episode_sha256,
                    time.time(),
                    battle_tag,
                    worker_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"worker does not own leased task: {battle_tag}")

    def fail(
        self,
        worker_id: str,
        battle_tag: str,
        error: str,
        *,
        retry: bool = True,
    ) -> None:
        status = "pending" if retry else "failed"
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE tasks SET status=?, lease_owner=NULL, lease_until=NULL,
                    last_error=?, updated_at=?
                WHERE battle_tag=? AND status='leased' AND lease_owner=?
                """,
                (status, error[:4000], time.time(), battle_tag, worker_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"worker does not own leased task: {battle_tag}")

    def counts(self) -> dict[str, int]:
        return {
            str(row["status"]): int(row["count"])
            for row in self.connection.execute(
                "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
            )
        }

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "TickStoreWorkQueue":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

