from contextlib import contextmanager
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from xiaoying.storage.db import fence


class FencedPostgresSaver(PostgresSaver):
    """Use the stock schema/serialization, fence every storage operation in its transaction.

    _cursor is a version-pinned adapter seam (langgraph-checkpoint-postgres in uv.lock).
    No pipeline may commit checkpoint writes after the lease check transaction ends.
    """

    def __init__(self, conn, job_id, token):
        super().__init__(conn)
        self.job_id, self.token = job_id, token

    @contextmanager
    def _cursor(self, *, pipeline=False):
        with self.lock, self.conn.transaction():
            fence(self.conn, self.job_id, self.token)
            with self.conn.cursor(binary=True, row_factory=dict_row) as cursor:
                yield cursor
