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

from fastapi import FastAPI, Request, Response
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
# Routers
# ---------------------------------------------------------------------------

from app.api.auth import router as auth_router
from app.api.upload import router as upload_router
from app.api.search import router as search_router
from app.api.debug import router as debug_router
from app.api.conversation import router as conversation_router
from app.api.collection import router as collection_router
from app.api.retrieval import router as retrieval_router

app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(search_router)
app.include_router(debug_router)
app.include_router(conversation_router)
app.include_router(collection_router)
app.include_router(retrieval_router)

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


@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    """
    Render the landing page.

    GET  -> Returns the HTML page.
    HEAD -> Returns only HTTP headers.
    """

    if request.method == "HEAD":
        return Response(status_code=200)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "Enterprise Knowledge Intelligence Platform"},
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    """
    Render the Login page.
    """
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"title": "Sign In - Enterprise Knowledge Intelligence Platform"},
    )


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request) -> HTMLResponse:
    """
    Render the Registration page.
    """
    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={"title": "Register - Enterprise Knowledge Intelligence Platform"},
    )
