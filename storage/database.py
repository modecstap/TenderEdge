from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from .config import DbConfig
from .declarative_base import DeclarativeBase
from extensions.singleton import Singleton
from .utils import get_db_url


class Database(metaclass=Singleton):
    def __init__(self):
        db_config = DbConfig
        db_url = get_db_url(
            user=db_config.user,
            password=db_config.password,
            host=db_config.host,
            port=db_config.port,
            dname=db_config.dname
        )

        # создаем асинхронный движок для взаимодействия с БД
        self.async_engine = create_async_engine(db_url)

        # инициализируем асинхронную фабрику сессий
        self.async_Session = sessionmaker(self.async_engine, expire_on_commit=False, class_=AsyncSession)

    # создаем все необходимые таблицы, если они еще не созданы
    async def create_all(self):
        async with self.async_engine.begin() as conn:
            await conn.run_sync(DeclarativeBase().base.metadata.create_all)
