"""
app/main.py
-----------
FastAPI application entry point.

Phase 2: Initial Web Application

Configures:
    - Static file serving  (/static → app/static)
    - Jinja2 template rendering  (app/templates)
    - Root route  GET /  → renders templates/index.html
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Enterprise Knowledge Intelligence Platform",
    description="Production-ready AI-powered RAG Platform",
    version="0.2.0",
)

# ---------------------------------------------------------------------------
# Static files — served at /static
# ---------------------------------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static",
)

# ---------------------------------------------------------------------------
# Jinja2 templates
# ---------------------------------------------------------------------------

templates = Jinja2Templates(directory="app/templates")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Render the landing page / dashboard.

    Returns:
        HTMLResponse: Rendered index.html template.
    """
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "Enterprise Knowledge Intelligence Platform"},
    )
