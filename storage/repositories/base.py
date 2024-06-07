from storage.database import Database


class BaseRepository:
    def __init__(self):
        self.db = Database()

    async def upsert(self, entity):
        async with self.db.async_Session() as session:
            await session.merge(entity)
            await session.commit()
