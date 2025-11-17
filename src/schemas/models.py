"""Pydantic models for the construction materials bot system."""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class DeliveryInfo(BaseModel):
    """Delivery information."""
    address: Optional[str] = Field(None, description="Delivery address")
    date: Optional[str] = Field(None, description="Delivery date")


class ProductCharacteristics(BaseModel):
    """Product characteristics."""
    mark: Optional[str] = Field(None, description="Product mark (e.g., М300, М400)")
    fraction: Optional[str] = Field(None, description="Fraction size (e.g., 20-40, 5-20)")
    product_type: Optional[str] = Field(None, description="Additional product type specification")


class OrderSpecs(BaseModel):
    """Order specifications extracted from user query."""
    product_type: Optional[str] = Field(None, description="Type of product (бетон, песок, гравий, щебень)")
    quantity: Optional[str] = Field(None, description="Quantity (volume or weight, e.g., '5 кубов', '10 тонн')")
    characteristics: Optional[ProductCharacteristics] = Field(None, description="Product characteristics")
    delivery: Optional[DeliveryInfo] = Field(None, description="Delivery information")
    
    def is_complete(self) -> bool:
        """Check if order specifications are complete."""
        return (
            self.product_type is not None
            and self.quantity is not None
            and self.characteristics is not None
            and self.characteristics.mark is not None
        )
    
    def get_missing_fields(self) -> list[str]:
        """Get list of missing required fields."""
        missing = []
        if not self.product_type:
            missing.append("product_type")
        if not self.quantity:
            missing.append("quantity")
        if not self.characteristics or not self.characteristics.mark:
            missing.append("characteristics.mark")
        return missing


class UserQuery(BaseModel):
    """User query input."""
    message: str = Field(..., description="User message")
    session_id: str = Field(..., description="Session identifier for multi-turn dialogue")


class BotResponse(BaseModel):
    """Bot response output."""
    message: str = Field(..., description="Response message to user")
    needs_clarification: bool = Field(False, description="Whether clarification is needed")
    extracted_specs: Optional[OrderSpecs] = Field(None, description="Extracted order specifications")
    query_type: Literal["informational", "order_specification"] = Field(
        ..., description="Type of query"
    )

