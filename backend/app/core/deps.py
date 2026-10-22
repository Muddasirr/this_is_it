from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import decode_access_token
from app.models import FacilityUser, User
from app.models.entities import FacilityRole

bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    if creds is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = decode_access_token(creds.credentials)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = db.get(User, UUID(payload["sub"]))
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User inactive")
    return user


def require_facility_owner(
    facility_id: UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FacilityUser:
    membership = require_facility_member(facility_id, user, db)
    if membership.role != FacilityRole.owner:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Facility owner role required")
    return membership


def require_facility_member(
    facility_id: UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FacilityUser:
    membership = (
        db.query(FacilityUser)
        .filter(
            FacilityUser.facility_id == facility_id,
            FacilityUser.user_id == user.id,
        )
        .first()
    )
    if not membership:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a facility member")
    return membership
