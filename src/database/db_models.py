from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, Index
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Store(Base):
    __tablename__ = "stores"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True, index=True)
    address = Column(String, nullable=True)

    products = relationship("Product", back_populates="store")


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)

    name = Column(String, nullable=False)
    category = Column(String, nullable=True, index=True)
    description = Column(String, nullable=True)

    model = Column(String, nullable=True, index=True)
    fraction = Column(String, nullable=True, index=True)

    price = Column(Float, nullable=True)
    unit = Column(String, default="kg")
    stock = Column(Integer, default=0)

    store_id = Column(Integer, ForeignKey("stores.id"), index=True)
    store = relationship("Store", back_populates="products")

    is_active = Column(Boolean, default=True)


Index("idx_category_model", Product.category, Product.model)
Index("idx_category_fraction", Product.category, Product.fraction)
