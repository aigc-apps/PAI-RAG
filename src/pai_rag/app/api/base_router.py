from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.get("/")
async def api_root():
    return RedirectResponse(url="/docs")


@router.get("/health")
def health_check():
    return {"status": "OK"}
