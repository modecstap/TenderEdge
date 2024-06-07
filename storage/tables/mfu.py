import json

from sqlalchemy import Column, String, Integer

from storage.declarative_base import DeclarativeBase

Base = DeclarativeBase().base


class Mfu(Base):
    __tablename__ = 'mfu'

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    vender = Column(String(50), nullable=False)
    functional = Column(String(50), nullable=False)
    price = Column(Integer, nullable=False)
    refueling_cost = Column(Integer, nullable=False)
    supplie_cost = Column(Integer, nullable=False)
    repairability = Column(String(50), nullable=False)
    parts_support = Column(String(50), nullable=False)
    manufacturer = Column(String(50), nullable=False)
    efficiency = Column(String(50), nullable=False)
    cluster = Column(Integer)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'vender': self.vender,
            'functional': self.functional,
            'price': self.price,
            'refueling_cost': self.refueling_cost,
            'supplie_cost': self.supplie_cost,
            'repairability': self.repairability,
            'parts_support': self.parts_support,
            'manufacturer': self.manufacturer,
            'efficiency': self.efficiency,
            'count_id': self.count_id
        }

    def to_json(self):
        return json.dumps(self.to_dict())
