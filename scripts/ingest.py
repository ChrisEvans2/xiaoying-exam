"""Same ingestion service as Worker; only pending database job IDs are accepted."""

import argparse
from xiaoying.cli import claim_specific
from xiaoying.config import load_settings
from xiaoying.storage.db import Database, require, valid_id
from xiaoying.worker import Worker

parser = argparse.ArgumentParser()
parser.add_argument("--job-id", required=True)
args = parser.parse_args()
settings = load_settings()
db = Database(settings.database_url)
try:
    record = require(
        db.one(
            "SELECT id FROM jobs WHERE id=%s AND type='build' AND status='pending'", (valid_id(args.job_id),)
        )
    )
    job = claim_specific(db, settings, str(record["id"]))
    if not job:
        raise SystemExit("任务已被其他 Worker 领取")
    Worker(db, settings).execute(job)
    result = db.one("SELECT status,error FROM jobs WHERE id=%s", (args.job_id,))
    print(result)
    if result["status"] != "done":
        raise SystemExit(1)
finally:
    db.close()
