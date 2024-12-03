from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.get("/")
async def api_root():
    return RedirectResponse(url="/docs")
