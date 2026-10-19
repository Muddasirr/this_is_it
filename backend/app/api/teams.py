from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import require_facility_member
from app.models import FacilityUser, Player, Team
from app.schemas import PlayerCreate, PlayerOut, TeamCreate, TeamOut
from app.services.avatars import avatar_url_for
from app.services.identity import rename_player

router = APIRouter()


class PlayerRename(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)


def _player_out(player: Player) -> PlayerOut:
    return PlayerOut(
        id=player.id,
        facility_id=player.facility_id,
        team_id=player.team_id,
        display_name=player.display_name,
        default_shirt_number=player.default_shirt_number,
        avatar_key=player.avatar_key,
        avatar_url=avatar_url_for(player),
        is_active=player.is_active,
        created_at=player.created_at,
    )


@router.post("/teams", response_model=TeamOut, status_code=status.HTTP_201_CREATED)
def create_team(
    facility_id: UUID,
    payload: TeamCreate,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> Team:
    team = Team(facility_id=facility_id, name=payload.name, primary_color=payload.primary_color)
    db.add(team)
    db.commit()
    db.refresh(team)
    return team


@router.get("/teams", response_model=list[TeamOut])
def list_teams(
    facility_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> list[Team]:
    return db.query(Team).filter(Team.facility_id == facility_id).order_by(Team.name).all()


@router.post("/players", response_model=PlayerOut, status_code=status.HTTP_201_CREATED)
def create_player(
    facility_id: UUID,
    payload: PlayerCreate,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> PlayerOut:
    if payload.team_id:
        team = db.get(Team, payload.team_id)
        if not team or team.facility_id != facility_id:
            raise HTTPException(status_code=400, detail="Team not in facility")
    player = Player(
        facility_id=facility_id,
        team_id=payload.team_id,
        display_name=payload.display_name,
        default_shirt_number=payload.default_shirt_number,
    )
    db.add(player)
    db.commit()
    db.refresh(player)
    return _player_out(player)


@router.get("/players", response_model=list[PlayerOut])
def list_players(
    facility_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> list[PlayerOut]:
    rows = (
        db.query(Player)
        .filter(Player.facility_id == facility_id, Player.is_active.is_(True))
        .order_by(Player.display_name)
        .all()
    )
    return [_player_out(p) for p in rows]


@router.patch("/players/{player_id}", response_model=PlayerOut)
def update_player_name(
    facility_id: UUID,
    player_id: UUID,
    payload: PlayerRename,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> PlayerOut:
    player = db.get(Player, player_id)
    if not player or player.facility_id != facility_id:
        raise HTTPException(status_code=404, detail="Player not found")
    renamed = rename_player(db, player_id, payload.display_name)
    return _player_out(renamed)
