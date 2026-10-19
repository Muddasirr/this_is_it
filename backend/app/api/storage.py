from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from app.services.storage import local_path, save_local_bytes, use_local_storage, verify_local_signature

router = APIRouter()


@router.put("/upload")
async def local_upload(
    request: Request,
    key: str = Query(...),
    expires: int = Query(...),
    signature: str = Query(...),
    content_type: str = Query("video/mp4"),
) -> dict:
    if not use_local_storage():
        raise HTTPException(status_code=404, detail="Local storage disabled")
    if not verify_local_signature(key, expires, signature):
        raise HTTPException(status_code=403, detail="Invalid or expired signature")
    data = await request.body()
    save_local_bytes(key, data)
    return {"ok": True, "key": key, "bytes": len(data), "content_type": content_type}


@router.get("/download")
def local_download(
    key: str = Query(...),
    expires: int = Query(...),
    signature: str = Query(...),
) -> Response:
    if not use_local_storage():
        raise HTTPException(status_code=404, detail="Local storage disabled")
    if not verify_local_signature(key, expires, signature):
        raise HTTPException(status_code=403, detail="Invalid or expired signature")
    path = local_path(key)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="video/mp4", filename=path.name)
