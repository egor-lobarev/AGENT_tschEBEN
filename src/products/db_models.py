from sqlalchemy import Column, Integer, String, Text, ForeignKey, Numeric, TIMESTAMP
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime

Base = declarative_base()

class Store(Base):
    __tablename__ = "stores"

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    address = Column(Text)
    phone = Column(Text)

    products = relationship("StoreProduct", back_populates="store")

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    category = Column(Text, nullable=False)
    model = Column(Text, nullable=False)
    name = Column(Text, nullable=False)
    attributes = Column(JSONB, default={})

    offers = relationship("StoreProduct", back_populates="product")

class StoreProduct(Base):
    __tablename__ = "store_products"

    id = Column(Integer, primary_key=True)
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    price = Column(Numeric(12, 2), nullable=False)
    stock = Column(Integer, nullable=False)
    unit = Column(Text, nullable=False)
    updated_at = Column(TIMESTAMP, default=datetime.now)

    store = relationship("Store", back_populates="products")
    product = relationship("Product", back_populates="offers")
