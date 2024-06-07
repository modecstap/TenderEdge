from sqlalchemy.orm import declarative_base

from extensions.singleton import Singleton


class DeclarativeBase(metaclass=Singleton):
    def __init__(self):
        self.base = declarative_base()
