"""
Mock API for products database.
This is a mock implementation of Лизы's product database API.
Can be easily replaced with a real API implementation.

To connect a real database API:
1. Replace the get_products() function with your implementation
2. Ensure the return format matches: List[Dict[str, Any]] with keys:
   - id, name, product_type, price_per_unit, unit, available, description
   - mark (optional), fraction (optional)
3. See README.md for detailed instructions
"""

from typing import List, Dict, Any
from src.schemas.models import OrderSpecs


def get_products(specs: OrderSpecs) -> List[Dict[str, Any]]:
    """
    Get products from database based on order specifications.
    This is the API function for Лизы's product database module.
    
    Args:
        specs: Order specifications
        
    Returns:
        List of product dictionaries matching the specifications
    """
    # Mock product database
    mock_products = [
        {
            "id": 1,
            "name": "Бетон М300",
            "product_type": "бетон",
            "mark": "М300",
            "price_per_unit": 3500,
            "unit": "куб.м",
            "available": True,
            "description": "Бетон марки М300, подходит для фундаментов и монолитных конструкций"
        },
        {
            "id": 2,
            "name": "Бетон М350",
            "product_type": "бетон",
            "mark": "М350",
            "price_per_unit": 3800,
            "unit": "куб.м",
            "available": True,
            "description": "Бетон марки М350, повышенная прочность для ответственных конструкций"
        },
        {
            "id": 3,
            "name": "Бетон М400",
            "product_type": "бетон",
            "mark": "М400",
            "price_per_unit": 4200,
            "unit": "куб.м",
            "available": True,
            "description": "Бетон марки М400, высокопрочный для промышленных объектов"
        },
        {
            "id": 4,
            "name": "Песок карьерный",
            "product_type": "песок",
            "mark": None,
            "fraction": "0-5",
            "price_per_unit": 800,
            "unit": "куб.м",
            "available": True,
            "description": "Карьерный песок, фракция 0-5 мм, для строительных работ"
        },
        {
            "id": 5,
            "name": "Песок речной",
            "product_type": "песок",
            "mark": None,
            "fraction": "0-2",
            "price_per_unit": 1200,
            "unit": "куб.м",
            "available": True,
            "description": "Речной песок, фракция 0-2 мм, высокое качество"
        },
        {
            "id": 6,
            "name": "Щебень гранитный 20-40",
            "product_type": "щебень",
            "mark": None,
            "fraction": "20-40",
            "price_per_unit": 2500,
            "unit": "куб.м",
            "available": True,
            "description": "Гранитный щебень фракции 20-40 мм, для бетона и дорожных работ"
        },
        {
            "id": 7,
            "name": "Щебень гранитный 5-20",
            "product_type": "щебень",
            "mark": None,
            "fraction": "5-20",
            "price_per_unit": 2800,
            "unit": "куб.м",
            "available": True,
            "description": "Гранитный щебень фракции 5-20 мм, мелкая фракция"
        },
        {
            "id": 8,
            "name": "Гравий 20-40",
            "product_type": "гравий",
            "mark": None,
            "fraction": "20-40",
            "price_per_unit": 1800,
            "unit": "куб.м",
            "available": True,
            "description": "Гравий фракции 20-40 мм, природный материал"
        }
    ]
    
    # Filter products based on specifications
    filtered_products = []
    
    for product in mock_products:
        match = True
        
        # Filter by product type
        if specs.product_type:
            if product["product_type"].lower() != specs.product_type.lower():
                match = False
                continue
        
        # Filter by mark (for бетон)
        if specs.characteristics and specs.characteristics.mark:
            if product.get("mark") and product["mark"] != specs.characteristics.mark:
                match = False
                continue
        
        # Filter by fraction (for щебень, гравий, песок)
        if specs.characteristics and specs.characteristics.fraction:
            if product.get("fraction") and product["fraction"] != specs.characteristics.fraction:
                match = False
                continue
        
        if match:
            filtered_products.append(product)
    
    # If no products match exactly, return products matching at least product_type
    if not filtered_products and specs.product_type:
        for product in mock_products:
            if product["product_type"].lower() == specs.product_type.lower():
                filtered_products.append(product)
    
    # If still no products, return all products (fallback)
    if not filtered_products:
        filtered_products = mock_products[:3]  # Return first 3 as examples
    
    return filtered_products

