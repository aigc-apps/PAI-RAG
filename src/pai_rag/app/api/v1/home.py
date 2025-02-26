import os
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


_BASE_DIR = Path(__file__).parent

router_home = APIRouter()


# Set up Jinja2 template rendering
templates = Jinja2Templates(directory=os.path.join(_BASE_DIR, "templates"))


@router_home.get("", response_class=HTMLResponse)
async def homepage(request: Request):
    # Render the index.html template with a message variable
    return templates.TemplateResponse(
        "index.html", {"request": request, "message": "Hi, welcome to PAI-RAG!"}
    )
