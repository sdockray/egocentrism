"""
Entrypoint. This is intentionally a stub — drop your existing
segment/extract/reduce logic in as sibling modules (download.py,
segment.py, features.py, reduce.py) and call them from here, using
src.storage for reads/writes and src.db.start_run to log provenance.
"""
from src.db import start_run


def main():
    run_id = start_run(config={"stage": "smoke_test"}, notes="scaffold smoke test")
    print(f"Started run {run_id}. Wire up download/segment/features/reduce steps here.")


if __name__ == "__main__":
    main()
