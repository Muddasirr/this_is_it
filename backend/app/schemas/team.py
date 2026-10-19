from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class TeamCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    primary_color: str | None = None


class TeamOut(BaseModel):
    id: UUID
    facility_id: UUID
    name: str
    primary_color: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class PlayerCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    team_id: UUID | None = None
    default_shirt_number: int | None = Field(default=None, ge=0, le=99)


class PlayerOut(BaseModel):
    id: UUID
    facility_id: UUID
    team_id: UUID | None
    display_name: str
    default_shirt_number: int | None
    avatar_key: str | None = None
    avatar_url: str | None = None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}
