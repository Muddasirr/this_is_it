from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.entities import FacilityRole


class FacilityCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    timezone: str = "UTC"


class FacilityOut(BaseModel):
    id: UUID
    name: str
    timezone: str
    created_at: datetime

    model_config = {"from_attributes": True}


class FacilityUserOut(BaseModel):
    id: UUID
    facility_id: UUID
    user_id: UUID
    role: FacilityRole
    created_at: datetime

    model_config = {"from_attributes": True}
