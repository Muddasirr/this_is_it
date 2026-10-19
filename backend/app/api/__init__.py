from fastapi import APIRouter

from app.api import auth, facilities, matches, phase0, review, share, storage, teams, vision

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(storage.router, prefix="/storage", tags=["storage"])
api_router.include_router(phase0.router, prefix="/phase0", tags=["phase0"])
api_router.include_router(share.router, prefix="/share", tags=["share"])
api_router.include_router(facilities.router, prefix="/facilities", tags=["facilities"])
api_router.include_router(teams.router, prefix="/facilities/{facility_id}", tags=["teams"])
api_router.include_router(matches.router, prefix="/facilities/{facility_id}/matches", tags=["matches"])
api_router.include_router(vision.router, prefix="/facilities/{facility_id}/matches", tags=["vision"])
api_router.include_router(review.router, prefix="/facilities/{facility_id}/matches", tags=["review"])
