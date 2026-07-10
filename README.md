# Enterprise Knowledge Intelligence Platform (EKIP)
### Placement Edition v1.0

> A production-ready, AI-powered Retrieval-Augmented Generation (RAG) platform built with FastAPI, ChromaDB, Google Gemini 2.5 Flash, and vanilla HTML/CSS/JS.

---

## Overview

EKIP is a full-stack enterprise knowledge management system that transforms static document repositories into a living, conversational intelligence layer. Users upload enterprise documents (PDF, DOCX, TXT), which are processed, chunked, and embedded into a ChromaDB vector store. A streaming RAG pipeline then enables natural-language conversations grounded in the actual document content.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI 0.138, Python 3.11, SQLAlchemy 2.0 |
| **AI / LLM** | Google Gemini 2.5 Flash (`gemini-2.5-flash`) |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Vector Store** | ChromaDB (local persistent store) |
| **Database** | SQLite (default) / PostgreSQL (production) |
| **Auth** | JWT (python-jose), bcrypt password hashing |
| **Frontend** | Jinja2 templates, Bootstrap 5, Vanilla JS |
| **Document Parsing** | PyMuPDF (PDF), python-docx (DOCX) |

---

## Features

- **User Authentication** — JWT-based login/register/logout with route protection
- **Document Management** — Upload, process, chunk, and vector-index PDF, DOCX, and TXT files
- **Streaming RAG** — Real-time streaming responses via Server-Sent Events (SSE) with Gemini 2.5 Flash
- **Conversation Management** — Multi-turn conversations with memory, pinning, archiving, rename, and soft-delete
- **Conversation Memory** — Up to 10 prior exchange pairs are injected into each RAG prompt for context continuity
- **Semantic Search** — Standalone vector search endpoint with structured citations and LLM-generated answers
- **Search History** — Logged RAG query history with latency metrics, trend charts, and frequency analysis
- **Dashboard** — Real-time metrics, system details, and recent activity stream
- **Activity Logging** — Automatic logging of document uploads, conversation events, and RAG queries

---

## Project Structure

```
Enterprise-Knowledge-Intelligence-Platform/
├── app/
│   ├── ai/                         # Gemini 2.5 Flash service & prompt builder
│   ├── api/                        # FastAPI route handlers
│   │   ├── auth.py                 # Login, register, logout, JWT validation
│   │   ├── conversation.py         # CRUD + streaming RAG endpoint
│   │   ├── dashboard.py            # Dashboard metrics API
│   │   ├── retrieval.py            # Collection-scoped semantic retrieval
│   │   ├── search.py               # Standalone semantic search
│   │   ├── search_history.py       # Search history CRUD & analytics
│   │   └── upload.py               # Document upload, process, chunk
│   ├── database/                   # SQLAlchemy engine, repositories
│   ├── models/                     # ORM models (User, Document, Chunk, Conversation, etc.)
│   ├── repositories/               # Data access layer
│   ├── schemas/                    # Pydantic request/response schemas
│   ├── services/                   # Business logic layer
│   ├── static/                     # CSS, JS assets
│   ├── templates/                  # Jinja2 HTML templates
│   └── utils/                      # Activity logger, ChromaDB cleaner
├── chroma_db/                      # Local ChromaDB vector store (gitignored in production)
├── uploads/                        # Uploaded document files (gitignored in production)
├── tests/                          # Test scripts
├── .env.example                    # Environment variable template
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Google Gemini API Key](https://aistudio.google.com/app/apikey)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/srinidhiumasankar/Enterprise-Knowledge-Intelligence-Platform.git
cd Enterprise-Knowledge-Intelligence-Platform

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env     # Windows
# cp .env.example .env     # Linux/macOS

# Edit .env and add your GEMINI_API_KEY
```

### Running the Application

```bash
uvicorn app.main:app --reload
```

Open your browser at: **http://localhost:8000**

### API Documentation (Swagger UI)

```
http://localhost:8000/docs
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key for RAG generation |
| `SECRET_KEY` | ✅ Yes | JWT signing secret (use a long random string) |
| `APP_ENV` | No | `development` / `staging` / `production` |
| `UPLOAD_DIR` | No | Directory for uploaded files (default: `uploads`) |
| `MAX_UPLOAD_SIZE_MB` | No | Max upload file size in MB (default: `20`) |
| `DATABASE_URL` | No | PostgreSQL URL (default: SQLite `ekip.db`) |

---

## Usage Workflow

1. **Register** an account at `/register`
2. **Login** at `/login`
3. **Upload** enterprise documents (PDF, DOCX, TXT) via the Documents page
4. **Process** each document — extracts text and generates vector embeddings
5. **Start a Conversation** — ask natural-language questions about your documents
6. **View Search History** — review logged RAG query metrics and trends
7. **Check Dashboard** — monitor system metrics and recent activity

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/register` | Register a new user |
| `POST` | `/api/auth/login` | Login and receive JWT |
| `POST` | `/api/auth/logout` | Invalidate session |
| `GET` | `/api/auth/me` | Get authenticated user profile |
| `POST` | `/api/upload` | Upload a document |
| `GET` | `/api/upload` | List user documents |
| `POST` | `/api/upload/{id}/process` | Process and chunk document |
| `DELETE` | `/api/upload/{id}` | Delete a document |
| `POST` | `/api/search` | Semantic search |
| `GET` | `/api/conversations` | List conversations |
| `POST` | `/api/conversations` | Create conversation |
| `POST` | `/api/conversations/{id}/stream` | **Streaming RAG** (SSE) |
| `PATCH` | `/api/conversations/{id}/rename` | Rename conversation |
| `PATCH` | `/api/conversations/{id}/pin` | Pin conversation |
| `PATCH` | `/api/conversations/{id}/archive` | Archive conversation |
| `DELETE` | `/api/conversations/{id}` | Soft-delete conversation |
| `GET` | `/api/dashboard` | Dashboard metrics |
| `GET` | `/api/search-history` | Paginated query history |
| `GET` | `/api/search-history/statistics` | Aggregated search stats |
| `GET` | `/api/search-history/frequent` | Top searched queries |

---

## Architecture

```
User → Browser (Jinja2 + Vanilla JS)
         │
         ▼
    FastAPI Router
         │
    ┌────┴────────────────────────┐
    │                             │
  Auth & Session           RAG Pipeline
    │                             │
  SQLite/PostgreSQL         1. Query Embedding (Sentence-Transformers)
  (Users, Conversations,    2. Vector Retrieval (ChromaDB)
   Documents, Chunks,       3. Prompt Construction (PromptBuilder)
   Search History)          4. LLM Generation (Gemini 2.5 Flash)
                            5. Streaming SSE Response → Browser
```

---

## Development Notes

- The ChromaDB vector store is persisted locally in `chroma_db/`. Do not commit this directory to production.
- Document files are stored in `uploads/`. Do not commit this directory to production.
- The SQLite database (`ekip.db`) is auto-created on first startup via `Base.metadata.create_all()`.
- All API routes require JWT authentication except `/api/auth/register` and `/api/auth/login`.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built as an enterprise portfolio project demonstrating full-stack AI engineering capabilities.*
