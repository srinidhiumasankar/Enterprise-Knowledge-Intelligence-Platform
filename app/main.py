"""
app/main.py
-----------
FastAPI application entry point for the Enterprise Knowledge Intelligence Platform.

Placement Edition v1.0

Configures:
    - Static file serving  (/static → app/static)
    - Jinja2 template rendering  (app/templates)
    - API routers for auth, upload, search, conversations, retrieval, dashboard
    - Database auto-migration on startup
    - ChromaDB stale vector cleanup on startup
"""
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Enterprise Knowledge Intelligence Platform",
    description="Production-ready AI-powered Retrieval-Augmented Generation (RAG) Platform — Placement Edition v1.0",
    version="1.0.0",
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
from app.api.workspace_context import router as workspace_context_router
from app.api.workspace import router as workspace_router
from app.api.search_history import router as search_history_router
from app.api.dashboard import router as dashboard_router

app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(search_router)
app.include_router(debug_router)
app.include_router(conversation_router)
app.include_router(collection_router)
app.include_router(retrieval_router)
app.include_router(workspace_context_router)
app.include_router(workspace_router)
app.include_router(search_history_router)
app.include_router(dashboard_router)

@app.on_event("startup")
def startup_event():
    from app.database.connection import engine
    from app.database.base import Base
    import app.models  # ensure models are registered
    Base.metadata.create_all(bind=engine)
    try:
        from app.utils.chroma_cleaner import cleanup_stale_vectors
        cleanup_stale_vectors()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Startup ChromaDB cleanup failed: {e}")

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


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> HTMLResponse:
    """
    Render the Workspace Dashboard UI page.
    """
    from app.config import settings
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "title": "Workspace Dashboard - Enterprise Knowledge Intelligence Platform",
            "settings": settings
        },
    )


@app.get("/conversations", response_class=HTMLResponse)
async def conversations_page(request: Request) -> HTMLResponse:
    """
    Render the Conversations UI page.
    """
    return templates.TemplateResponse(
        request=request,
        name="conversations.html",
        context={"title": "Conversations - Enterprise Knowledge Intelligence Platform"},
    )


@app.get("/collections")
async def collections_redirect() -> RedirectResponse:
    """
    Collections have been removed from the Placement Edition UI.
    Redirect any direct navigation to the dashboard.
    """
    return RedirectResponse(url="/dashboard", status_code=301)


@app.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request) -> HTMLResponse:
    """
    Render the Documents UI page.
    """
    return templates.TemplateResponse(
        request=request,
        name="documents.html",
        context={"title": "Documents - Enterprise Knowledge Intelligence Platform"},
    )


@app.get("/workspaces")
async def workspaces_page() -> RedirectResponse:
    """
    Redirect the workspaces UI page to the dashboard.
    """
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get("/search-history", response_class=HTMLResponse)
async def search_history_page(request: Request) -> HTMLResponse:
    """
    Render the Search History UI page.
    """
    return templates.TemplateResponse(
        request=request,
        name="search_history.html",
        context={"title": "Search History - Enterprise Knowledge Intelligence Platform"},
    )
