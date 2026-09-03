"""Exactly-once PPO batch and rollout-shard ledger."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import time


BATCH_STATES = (
    "OPEN", "COLLECTING", "CLOSED", "UPDATING", "VALIDATING", "COMMITTED"
)
TRANSITIONS = {
    "OPEN": {"COLLECTING"},
    "COLLECTING": {"CLOSED"},
    "CLOSED": {"UPDATING"},
    "UPDATING": {"VALIDATING", "CLOSED"},
    "VALIDATING": {"COMMITTED", "CLOSED"},
    "COMMITTED": set(),
}


class RolloutLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS batches(
          batch_id TEXT PRIMARY KEY,
          policy_version INTEGER NOT NULL,
          actor_sha256 TEXT NOT NULL,
          state TEXT NOT NULL,
          created_at REAL NOT NULL,
          updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS shards(
          shard_uuid TEXT PRIMARY KEY,
          batch_id TEXT NOT NULL REFERENCES batches(batch_id),
          content_sha256 TEXT NOT NULL,
          consumed INTEGER NOT NULL DEFAULT 0
        );
        """)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def open_batch(self, batch_id: str, *, policy_version: int, actor_sha256: str) -> None:
        if not batch_id or policy_version < 0 or len(actor_sha256) != 64:
            raise ValueError("invalid batch identity")
        now = time.time()
        with self.connection:
            self.connection.execute(
                "INSERT INTO batches VALUES(?,?,?,?,?,?)",
                (batch_id, policy_version, actor_sha256, "OPEN", now, now),
            )

    def state(self, batch_id: str) -> str:
        row = self.connection.execute(
            "SELECT state FROM batches WHERE batch_id=?", (batch_id,)
        ).fetchone()
        if row is None:
            raise KeyError(batch_id)
        return str(row[0])

    def transition(self, batch_id: str, target: str) -> None:
        current = self.state(batch_id)
        if target not in TRANSITIONS[current]:
            raise RuntimeError(f"invalid batch transition {current}->{target}")
        with self.connection:
            changed = self.connection.execute(
                "UPDATE batches SET state=?,updated_at=? WHERE batch_id=? AND state=?",
                (target, time.time(), batch_id, current),
            ).rowcount
            if changed != 1:
                raise RuntimeError("concurrent batch transition conflict")

    def record_shard(self, batch_id: str, *, shard_uuid: str, content_sha256: str) -> bool:
        if self.state(batch_id) != "COLLECTING":
            raise RuntimeError("rollout shard arrived outside COLLECTING")
        if not shard_uuid or len(content_sha256) != 64:
            raise ValueError("invalid rollout shard identity")
        existing = self.connection.execute(
            "SELECT batch_id,content_sha256 FROM shards WHERE shard_uuid=?",
            (shard_uuid,),
        ).fetchone()
        if existing is not None:
            if tuple(existing) != (batch_id, content_sha256):
                raise RuntimeError("duplicate shard UUID has conflicting content")
            return False
        with self.connection:
            self.connection.execute(
                "INSERT INTO shards(shard_uuid,batch_id,content_sha256) VALUES(?,?,?)",
                (shard_uuid, batch_id, content_sha256),
            )
        return True

    def close_collection(self, batch_id: str, *, minimum_shards: int = 1) -> list[str]:
        rows = self.connection.execute(
            "SELECT shard_uuid FROM shards WHERE batch_id=? ORDER BY shard_uuid",
            (batch_id,),
        ).fetchall()
        if len(rows) < minimum_shards:
            raise RuntimeError("cannot close an incomplete rollout batch")
        self.transition(batch_id, "CLOSED")
        return [str(row[0]) for row in rows]

    def commit(self, batch_id: str) -> None:
        if self.state(batch_id) != "VALIDATING":
            raise RuntimeError("only a validated PPO batch can commit")
        with self.connection:
            changed = self.connection.execute(
                "UPDATE batches SET state='COMMITTED',updated_at=? "
                "WHERE batch_id=? AND state='VALIDATING'",
                (time.time(), batch_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("batch commit conflict")
            self.connection.execute(
                "UPDATE shards SET consumed=1 WHERE batch_id=?", (batch_id,)
            )

    def shards(self, batch_id: str) -> list[tuple[str, str, bool]]:
        return [
            (str(uuid), str(digest), bool(consumed))
            for uuid, digest, consumed in self.connection.execute(
                "SELECT shard_uuid,content_sha256,consumed FROM shards "
                "WHERE batch_id=? ORDER BY shard_uuid", (batch_id,)
            )
        ]
