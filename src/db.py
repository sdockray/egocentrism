import os
import subprocess
import json
from contextlib import contextmanager

from sqlalchemy import create_engine, text

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(os.environ["DATABASE_URL"])
    return _engine


@contextmanager
def get_conn():
    with get_engine().connect() as conn:
        yield conn


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        return None


def start_run(config: dict, notes: str = "") -> str:
    """Log a new pipeline run and return its run_id. Call this once at the
    start of any script that writes segments/features/reductions, and
    stamp every row you write with the returned run_id — this is what
    lets you trace any datapoint back to the exact code + config that
    produced it."""
    with get_conn() as conn:
        result = conn.execute(
            text(
                "INSERT INTO runs (git_commit, config_json, notes) "
                "VALUES (:git_commit, :config_json, :notes) RETURNING run_id"
            ),
            {
                "git_commit": current_git_commit(),
                "config_json": json.dumps(config),
                "notes": notes,
            },
        )
        conn.commit()
        return str(result.scalar())
