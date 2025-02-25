from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.get("/docs")
async def api_root():
    return RedirectResponse(url="/docs")
