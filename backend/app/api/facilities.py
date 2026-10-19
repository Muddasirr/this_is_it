from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.core.deps import get_current_user, require_facility_member
from app.models import Facility, FacilityUser, RetentionPolicy, User, VideoAsset
from app.models.entities import FacilityRole, VideoKind
from app.schemas import FacilityCreate, FacilityOut, FacilityUserOut
from app.services.storage import delete_object

router = APIRouter()


class RetentionOut(BaseModel):
    facility_id: UUID
    original_days: int
    proxy_days: int
    clip_days: int

    model_config = {"from_attributes": True}


class RetentionUpdate(BaseModel):
    original_days: int = Field(ge=1, le=3650)
    proxy_days: int = Field(ge=1, le=3650)
    clip_days: int = Field(ge=1, le=3650)


@router.post("", response_model=FacilityOut, status_code=status.HTTP_201_CREATED)
def create_facility(
    payload: FacilityCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Facility:
    facility = Facility(name=payload.name, timezone=payload.timezone)
    db.add(facility)
    db.flush()
    membership = FacilityUser(facility_id=facility.id, user_id=user.id, role=FacilityRole.owner)
    db.add(membership)
    db.add(
        RetentionPolicy(
            facility_id=facility.id,
            original_days=get_settings().default_retention_days,
        )
    )
    db.commit()
    db.refresh(facility)
    return facility


@router.get("", response_model=list[FacilityOut])
def list_facilities(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[Facility]:
    return (
        db.query(Facility)
        .join(FacilityUser)
        .filter(FacilityUser.user_id == user.id)
        .order_by(Facility.created_at.desc())
        .all()
    )


@router.get("/{facility_id}", response_model=FacilityOut)
def get_facility(
    facility_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> Facility:
    return db.get(Facility, facility_id)


@router.get("/{facility_id}/members", response_model=list[FacilityUserOut])
def list_members(
    facility_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> list[FacilityUser]:
    return db.query(FacilityUser).filter(FacilityUser.facility_id == facility_id).all()


@router.get("/{facility_id}/retention", response_model=RetentionOut)
def get_retention(
    facility_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> RetentionPolicy:
    policy = db.query(RetentionPolicy).filter(RetentionPolicy.facility_id == facility_id).first()
    if not policy:
        raise HTTPException(status_code=404, detail="Retention policy missing")
    return policy


@router.put("/{facility_id}/retention", response_model=RetentionOut)
def update_retention(
    facility_id: UUID,
    payload: RetentionUpdate,
    db: Session = Depends(get_db),
    membership: FacilityUser = Depends(require_facility_member),
) -> RetentionPolicy:
    if membership.role not in {FacilityRole.owner, FacilityRole.staff}:
        raise HTTPException(status_code=403, detail="Insufficient role")
    policy = db.query(RetentionPolicy).filter(RetentionPolicy.facility_id == facility_id).first()
    if not policy:
        policy = RetentionPolicy(facility_id=facility_id)
        db.add(policy)
    policy.original_days = payload.original_days
    policy.proxy_days = payload.proxy_days
    policy.clip_days = payload.clip_days
    db.commit()
    db.refresh(policy)
    return policy


@router.post("/{facility_id}/retention/enforce")
def enforce_retention(
    facility_id: UUID,
    db: Session = Depends(get_db),
    membership: FacilityUser = Depends(require_facility_member),
) -> dict:
    """Delete expired video assets for this facility per retention policy."""
    if membership.role != FacilityRole.owner:
        raise HTTPException(status_code=403, detail="Owner only")
    policy = db.query(RetentionPolicy).filter(RetentionPolicy.facility_id == facility_id).first()
    if not policy:
        raise HTTPException(status_code=404, detail="Retention policy missing")

    from app.models import Match

    matches = db.query(Match).filter(Match.facility_id == facility_id).all()
    match_ids = [m.id for m in matches]
    deleted = 0
    now = datetime.now(UTC)
    kind_days = {
        VideoKind.original: policy.original_days,
        VideoKind.proxy: policy.proxy_days,
        VideoKind.clip: policy.clip_days,
    }
    if match_ids:
        assets = db.query(VideoAsset).filter(VideoAsset.match_id.in_(match_ids)).all()
        for asset in assets:
            days = kind_days.get(asset.kind)
            if days is None:
                continue
            created = asset.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            if created < now - timedelta(days=days):
                try:
                    delete_object(asset.object_key)
                except Exception:  # noqa: BLE001
                    pass
                db.delete(asset)
                deleted += 1
        db.commit()
    return {"deleted_assets": deleted}
