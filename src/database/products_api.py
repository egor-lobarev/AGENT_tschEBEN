from sqlalchemy import select
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from .db_models import Product, Store, Base
from src.schemas.models import OrderSpecs


class ProductDatabase:
    def __init__(self, db_url: str, echo: bool = False):
        self.engine = create_engine(db_url, echo=echo)
        self.init_db()

    def init_db(self):
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

    def seed_from_json(self, seeds_dir: str):
        import os
        import json
        from sqlalchemy.orm import Session
        from .db_models import Store, Product

        with Session(self.engine) as session:
            for filename in os.listdir(seeds_dir):
                if not filename.endswith(".json"):
                    continue

                path = os.path.join(seeds_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                store_info = data["store"]
                store = Store(
                    name=store_info["name"],
                    address=store_info.get("address")
                )
                session.add(store)
                session.flush()

                for item in data["products"]:
                    product = Product(
                        name=item["name"],
                        category=item.get("category"),
                        model=item.get("model"),
                        fraction=item.get("fraction"),
                        description=item.get("description"),
                        price=item.get("price"),
                        stock=item.get("stock", 0),
                        unit=item.get("unit", "шт"),
                        store_id=store.id
                    )
                    session.add(product)

            session.commit()


    def get_products(self, specs: OrderSpecs, limit=50):
        with Session(self.engine) as session:
            if specs.product_type == 'None':
                print("Error of request. There is no category")

            stmt = select(Product).where(Product.is_active == True)

            if specs.product_type:
                stmt = stmt.where(Product.category == specs.product_type)

            if specs.characteristics:
                if specs.characteristics.mark:
                    stmt = stmt.where(Product.model == specs.characteristics.mark)

                if specs.characteristics.fraction:
                    stmt = stmt.where(Product.fraction == specs.characteristics.fraction)

            stmt = stmt.limit(limit)

            products = session.scalars(stmt).all()

            result = []
            for product in products:
                result.append({
                    "id": product.id,
                    "name": product.name,
                    "product_type": product.category,
                    "price_per_unit": float(product.price) if product.price else 0.0,
                    "unit": product.unit,
                    "available": product.stock > 0,
                    "description": product.description,
                    "mark": product.model,
                    "fraction": product.fraction,
                    "store": product.store.name if product.store else None,
                    "store_address": product.store.address if product.store else None,
                })

            return result