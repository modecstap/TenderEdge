from sqlalchemy import select

from ..repositories import BaseRepository
from ..tables.mfu import Mfu


class MfuRepository(BaseRepository):
    def __init__(self):
        super().__init__()

    async def get_all(self):
        async with self.db.async_Session() as session:
            result = await session.execute(
                select(Mfu)
            )
            return result.scalars().all()

    async def get_by_id(self, id: int):
        async with self.db.async_Session() as session:
            result = await session.execute(
                select(Mfu)
                .where(Mfu.id == id)
            )
            return result.scalar_one_or_none()

    async def get_cluster(self, cluster: int):
        async with self.db.async_Session() as session:
            result = await session.execute(
                select(Mfu)
                .where(Mfu.cluster == cluster)
            )
            return result.scalars().all()
