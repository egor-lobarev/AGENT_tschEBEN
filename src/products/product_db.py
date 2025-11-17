import json
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from .db_models import Base, Store, Product, StoreProduct


class ProductDatabase:
    def __init__(self, db_url: str, echo: bool = False):
        self.engine = create_engine(db_url, echo=echo)


    def init_db(self):
        """Создаёт таблицы, если их нет (использует модели SQLAlchemy)."""
        Base.metadata.create_all(self.engine)


    def seed_from_json(self, seeds_dir: str):
        """Загружает все json-файлы из seeds_dir, ожидается структура store -> products."""
        import os
        with Session(self.engine) as session:
            for f in os.listdir(seeds_dir):
                if not f.endswith('.json'):
                    continue
                path = os.path.join(seeds_dir, f)
                with open(path, 'r', encoding='utf-8') as fh:
                    store = json.load(fh)


            store_obj = Store(name=store['store_name'])
            session.add(store_obj)
            session.flush()


            for p in store['products']:
                existing = session.query(Product).filter_by(category=p['category'], model=p['model']).first()
                if not existing:
                    existing = Product(
                        category=p['category'],
                        model=p['model'],
                        name=p.get('name', f"{p['category']} {p['model']}"),
                        attributes=p.get('attributes', {})
                    )
                    session.add(existing)
                    session.flush()

                offer = StoreProduct(
                    store_id=store_obj.id,
                    product_id=existing.id,
                    price=p['price'],
                    stock=p['stock'],
                    unit=p.get('unit', 'шт')
                )
                session.add(offer)


            session.commit()


    def add_sample_data(self):
        """Быстрый сид (необязательно)."""
        sample = {
            'store_name': 'СтройМаркет №1',
            'products': [
                {'category': 'бетон', 'model': 'B20', 'name': 'Бетон B20', 'price': 4300, 'stock': 120, 'unit': 'м3', 'attributes': {'frost_resistance': 'F200'}},
                {'category': 'песок', 'model': 'мытый', 'name': 'Песок мытый', 'price': 630, 'stock': 210, 'unit': 'т'}
            ]
        }
        import tempfile, os
        td = tempfile.mkdtemp()
        path = os.path.join(td, 's.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(sample, fh, ensure_ascii=False)
        self.seed_from_json(td)


    def search(self, category=None, model=None, attributes: dict=None, limit=50):
        """Cтруктурированный поиск по товарам и атрибутам.
        Возвращает список StoreProduct (ORM объекты)"""
        with Session(self.engine) as session:
            q = session.query(StoreProduct).join(Product)
        if category:
            q = q.filter(Product.category == category)
        if model:
            q = q.filter(Product.model == model)
        if attributes:
            for k, v in attributes.items():
                q = q.filter(Product.attributes[k].astext == str(v))
        q = q.limit(limit)
        return q.all()


    def best_offer(self, category=None, model=None, attributes: dict=None):
        offers = self.search(category=category, model=model, attributes=attributes, limit=1000)
        if not offers:
            return None
        return min(offers, key=lambda o: float(o.price))