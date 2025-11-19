CREATE TABLE stores (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    address TEXT,
    phone TEXT
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    category TEXT NOT NULL,
    model TEXT NOT NULL,
    name TEXT NOT NULL,
    attributes JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE store_products (
    id SERIAL PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    price NUMERIC(12,2) NOT NULL,
    stock INTEGER NOT NULL,
    unit TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_model ON products(model);
CREATE INDEX idx_products_attributes ON products USING gin (attributes);
CREATE INDEX idx_store_products_product_id ON store_products(product_id);
