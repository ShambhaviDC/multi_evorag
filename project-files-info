# EvoRAG – Project Files & Architecture Documentation

This document provides a comprehensive inventory and functional explanation of all files created in the **EvoRAG (Evolutionary Multi-Personality Retrieval-Augmented Generation)** project.

---

## Table of Contents
1. [Project Overview & System Architecture](#project-overview--system-architecture)
2. [Project Directory Tree](#project-directory-tree)
3. [Core Retrieval & Consensus Engine](#1-core-retrieval--consensus-engine)
4. [FastAPI Backend, Authentication & Database](#2-fastapi-backend-authentication--database)
5. [Persona Evolution Engine](#3-persona-evolution-engine)
6. [Frontend & Browser Interface](#4-frontend--browser-interface)
7. [Startup & Orchestration Scripts](#5-startup--orchestration-scripts)
8. [Testing & Validation Testbenches](#6-testing--validation-testbenches)
9. [Data, Indexes & Storage](#7-data-indexes--storage)
10. [Documentation & Major Project Report](#8-documentation--major-project-report)

---

## Project Overview & System Architecture

EvoRAG is an advanced Retrieval-Augmented Generation framework featuring:
- **Phase 1 (Data Ingestion):** Automated news fetching across NewsAPI, GNews, and RSS feeds with cleaning, deduplication, and sliding-window chunking.
- **Phase 2 (Embedding & Vector Store):** High-speed semantic embedding using `sentence-transformers/all-MiniLM-L6-v2` and FAISS `IndexFlatL2` vector indexing with incremental atomic updates.
- **Phase 3 (Retrieval Pipeline):** Semantic candidate retrieval, diversity deduplication, prompt assembly, and sufficiency/freshness evaluation.
- **Phase 4 (Multi-Persona Engine):** Eight concurrent specialized reasoning personas analyzing the shared context.
- **Phase 5 (Peer Voting & Synthesis):** Multi-dimensional peer scoring matrix, dynamic score normalization, top-$K$ persona selection, and final consensus synthesis.
- **Phase 6 (Automated Knowledge Ingestion & Orchestration):** Background news scheduler (APScheduler, 3-day interval), dynamic hot-reloading of vectors in FastAPI, full Web Speech STT/TTS, and multi-turn chat persistence.
- **Continuous Persona Evolution:** Automated performance tracking, underperformer detection, prompt mutation, and hot-swapping.

---

## Project Directory Tree

```text
E:\Major\evorag_major
├── api.py                          # FastAPI REST API & static file server
├── auth.py                         # JWT authentication & bcrypt password hashing
├── build_ai_corpus.py              # Initial corpus construction helper
├── config.py                       # Global configuration constants & persona definitions
├── conversations.py                # Multi-turn chat session CRUD & message history
├── database.py                     # SQLite database engine & session dependency
├── embedder.py                     # Vector embedding, FAISS index & incremental updates
├── fetcher.py                      # NewsAPI, GNews & RSS article fetcher & chunker
├── generate_framework_diagram.py   # Architecture diagram generator
├── knowledge_scheduler.py          # 3-day automated news ingestion scheduler (APScheduler)
├── models.py                       # SQLModel database schemas (User, Conversation, Message)
├── personas.py                     # 8-Persona execution engine & prompt assembly
├── pipeline.py                     # Complete EvoRAG consensus orchestration pipeline
├── preferences.py                  # User topic & language preferences API
├── recommendations.py             # Personalized research query recommendations
├── retrieval_diagnostics.py        # Retrieval diversity & candidate distribution diagnostics
├── retriever.py                    # Semantic search, prompt builder, quality evaluation & auto-reload
├── run_app.ps1                     # PowerShell launcher for FastAPI & Web UI
├── run_scheduler.ps1               # PowerShell launcher for news ingestion scheduler
├── score_store.py                  # Peer voting history storage utility
├── score_store.json                # JSON persistence for peer evaluation scores
├── synthesis.py                    # Top-K persona selection & final answer synthesis
├── test_phase6_orchestration.py    # Automated test suite for Phase 6 orchestration & fallback
├── utils.py                        # Async LLM callers (Ollama, Gemini), retry logic & helpers
├── validate_phase1.py              # Phase 1 test bench (Fetch, clean, chunk)
├── validate_phase2.py              # Phase 2 test bench (FAISS build & query)
├── validate_phase3.py              # Phase 3 test bench (RAG retrieval & prompt)
├── validate_phase4.py              # Phase 4 test bench (8 personas generation)
├── validate_phase5.py              # Phase 5 test bench (Consensus voting & matrix)
├── voting.py                       # Multi-dimensional peer scoring & score matrix builder
├── requirements.txt                # Python package dependencies
├── .env / .env.example             # API keys and environment configurations
│
├── data/                           # Chunked news JSON datasets
│   ├── chunks_seed.json
│   └── chunks_*.json
│
├── index/                          # Persisted vector database files
│   ├── evorag.faiss                # FAISS binary vector index
│   ├── evorag_meta.json            # Chunk metadata sidecar
│   └── ingestion_status.json       # Scheduler update timestamps & vector metrics
│
├── frontend/                       # Web Client Application
│   ├── index.html                  # Main SPA interface (Consensus, Voice, Cards, Matrix, Chart)
│   ├── auth.js                     # Authentication token storage & login helpers
│   ├── styles.css                  # CSS theme & layout styling
│   ├── login.html                  # Standalone login page
│   ├── signup.html                 # Standalone registration page
│   ├── onboarding.html             # User topic preference onboarding
│   └── chat.html                   # Lightweight alternative chat view
│
├── persona_evolution/              # Evolutionary Auto-Tuning Engine
│   ├── tracker.py                  # Rolling average score tracking
│   ├── detector.py                 # Underperformer threshold detection
│   ├── mutator.py                  # Persona prompt mutation & generation
│   ├── replacer.py                 # In-memory hot-swapper & version tracker
│   └── __init__.py
│
└── evorag_phase2/                  # Major Project Academic Report (LaTeX & Diagrams)
    ├── main_report.tex             # Main LaTeX report wrapper
    ├── abstract.tex, introduction.tex, literature_review.tex, technical_description.tex, conclusion.tex
    ├── certificate.tex, acknowledgement.tex, bibliography.tex
    └── *.png / *.jpeg              # System architecture, HLD, LLD, and methodology diagrams
```

---

## 1. Core Retrieval & Consensus Engine

### `pipeline.py`
* **Purpose:** The central coordinator of the entire EvoRAG query-processing workflow.
* **Key Function:** `async def evorag_query(query: str, top_k: int) -> dict`
* **What it does:**
  1. Searches the local FAISS index for relevant candidate chunks.
  2. Calls `retriever.evaluate_retrieval()` to assess relevance (L2 distance), freshness (publication dates), and sufficiency (chunk count/words).
  3. If context is sufficient: invokes the 8-persona engine (`personas.py`), constructs the peer-evaluation score matrix (`voting.py`), selects the top-3 personas (`synthesis.py`), and synthesizes the final consensus answer.
  4. If context is insufficient/outdated: triggers the broad knowledge fallback engine, synthesizes distinct viewpoints for all 8 personas, generates a complete peer score matrix, and logs the query without revealing external API details.
  5. Records the query result and scores into `score_store.json`.

### `personas.py`
* **Purpose:** Multi-persona generation engine hosting the eight analytical viewpoints.
* **Key Functions:** `run_all_personas()`, `run_persona()`, `_get_response_with_fallback()`
* **What it does:**
  - Manages the 8 personas: `analytical-critical`, `empirical-evidential`, `adversarial-skeptical`, `synthesis-integrative`, `domain-expert`, `temporal-contextual`, `risk-evaluative`, and `creative-lateral`.
  - Concurrently prompts personas using an `asyncio.Semaphore` to manage local Ollama memory or calls Gemini asynchronously.
  - Automatically retries any persona that produces an empty or whitespace-only response.

### `voting.py`
* **Purpose:** Phase 5 Peer-Evaluation and Scoring Matrix Engine.
* **Key Functions:** `build_score_matrix()`, `compute_final_scores()`, `summarize_received_scores()`
* **What it does:**
  - Prompts each persona to rigorously evaluate every peer's response across three criteria: **factual grounding**, **reasoning quality**, and **completeness** (scores 1.0–10.0).
  - Normalizes scores, computes self-penalty adjustments, and builds the full $8 \times 8$ peer matrix for display on the UI heatmap.

### `synthesis.py`
* **Purpose:** Consensus synthesis and winner selection.
* **Key Functions:** `select_top_personas()`, `synthesize_top_k()`
* **What it does:**
  - Ranks all personas by their received peer scores and extracts the top-$K$ winners (default: top 3).
  - Prompts a dedicated synthesis model to merge the complementary strengths of the winning responses into a single, cohesive, authoritative consensus answer.

### `retriever.py`
* **Purpose:** Semantic search, prompt builder, quality evaluator, and index synchronizer.
* **Key Functions:** `retrieve()`, `evaluate_retrieval()`, `reload_index_if_updated()`, `build_prompt()`, `get_loaded_index_stats()`
* **What it does:**
  - Encodes user queries and searches FAISS vectors with candidate over-fetching and URL-level deduplication.
  - Formats context passages with numbered citations `[1]`, `[2]`.
  - **Dynamic Index Reloading:** Monitors disk file modification timestamps (`st_mtime`). Automatically reloads new vectors into memory when updated by the scheduler, avoiding API restarts.
  - Evaluates retrieval quality to determine whether local context is sufficient or if knowledge synthesis is needed.

### `embedder.py`
* **Purpose:** Phase 2 vector store management using `all-MiniLM-L6-v2`.
* **Key Functions:** `build_index()`, `load_index()`, `query()`, `add_chunks_to_index()`, `save_index_atomic()`
* **What it does:**
  - Generates 384-dimensional dense vector embeddings in batches using PyTorch.
  - Manages FAISS `IndexFlatL2` on disk (`index/evorag.faiss`) and JSON metadata sidecar (`index/evorag_meta.json`).
  - **Incremental Updates:** `add_chunks_to_index()` identifies newly discovered chunks using URL and SHA-256 text hashes, embeds only new chunks, appends them to FAISS via `.add()`, and performs atomic file swaps via `.tmp` files.

### `fetcher.py`
* **Purpose:** Phase 1 automated multi-source news data ingestion.
* **Key Functions:** `fetch()`, `_fetch_newsapi()`, `_fetch_gnews()`, `_fetch_rss()`, `_clean_text()`, `_build_chunks()`
* **What it does:**
  - Queries NewsAPI, GNews API, and 8 curated RSS feeds (BBC, Reuters, MIT Tech Review, Verge, Google AI, OpenAI, etc.).
  - Extracts full article bodies using BeautifulSoup, strips HTML boilerplate/scripts, deduplicates by URL hash, and splits text into overlapping sliding-window chunks (300–400 words).

### `knowledge_scheduler.py`
* **Purpose:** Phase 6 automated news ingestion scheduler.
* **Key Functions:** `is_update_due()`, `run_ingestion_pipeline()`, `start_scheduler()`
* **What it does:**
  - Implements APScheduler configured with a **3-day update frequency**.
  - On startup or trigger, verifies `ingestion_status.json`. If updated $< 3$ days ago, reuses the existing FAISS index.
  - If due, fetches recent articles across news sources, filters to new chunks, updates FAISS incrementally, and records timestamps and vector metrics.
  - Never executes personas during background runs to conserve CPU/RAM.

### `utils.py`
* **Purpose:** Shared network, timeout, and LLM communication helpers.
* **Key Functions:** `call_gemini_async()`, `call_ollama_async()`, `race_ollama_with_gemini()`, `word_count()`
* **What it does:**
  - Provides persistent `httpx.AsyncClient` connection pooling with keep-alive limits.
  - Implements exponential back-off retries for transient HTTP errors (429, 500, 502, 503).
  - Handles concurrent races between local Ollama and cloud APIs.

### `config.py`
* **Purpose:** Centralized project settings, tunable parameters, and model definitions.
* **What it defines:**
  - File paths (`ROOT_DIR`, `INDEX_DIR`, `DATA_DIR`, `SCORE_STORE_FILE`).
  - Model identifiers (`phi3:mini`, `all-MiniLM-L6-v2`, `gemini-flash-lite-latest`).
  - Persona system prompts, timeouts (`PERSONA_TIMEOUT_SECONDS = 300`), retrieval counts, and RSS feed URLs.

---

## 2. FastAPI Backend, Authentication & Database

### `api.py`
* **Purpose:** The primary REST API server and static frontend host.
* **Key Endpoints:**
  - `GET /`: Serves the primary web user interface (`frontend/index.html`).
  - `GET /health`: Returns server status, loaded FAISS vector count, and embedding dimensions.
  - `POST /query`: Single RAG query endpoint (Phase 3).
  - `POST /query/personas`: Parallel 8-persona response generation (Phase 4).
  - `POST /query/consensus`: Full consensus voting, peer matrix, and answer synthesis (Phase 5).
  - `POST /index/reload`: Manually triggers index synchronization from disk.
  - `GET /persona/status`: Returns persona health, rolling scores, and replacement history.
  - Mounts `/static` for assets and includes auth, conversation, preference, and recommendation routers.

### `auth.py`
* **Purpose:** User authentication, password hashing, and JWT token issuance.
* **Key Functions:** `get_password_hash()`, `verify_password()`, `create_access_token()`, `get_current_user()`
* **Endpoints:** `POST /auth/signup`, `POST /auth/token` (login), `GET /auth/me`.

### `models.py`
* **Purpose:** SQLModel (SQLAlchemy + Pydantic) database table schemas.
* **Tables Defined:**
  - `User`: Stored user credentials (`id`, `username`, `email`, `hashed_password`, `created_at`).
  - `Conversation`: Chat sessions (`id`, `user_id`, `title`, `created_at`, `updated_at`).
  - `Message`: Chat history messages (`id`, `conversation_id`, `role`, `content`, `timestamp`).
  - `UserPreference`: Stored user settings (`topics`, `expertise_level`, `preferred_persona`).

### `database.py`
* **Purpose:** Database engine setup and session lifecycle management.
* **Key Functions:** `create_db_and_tables()`, `get_session()`
* **What it does:** Connects to SQLite database `evorag.db` and injects database sessions into route handlers.

### `conversations.py`
* **Purpose:** Multi-turn chat session management and conversation persistence.
* **Key Functions:** `clean_title()` (auto-generates readable chat titles from queries), `verify_ownership()`.
* **Endpoints:**
  - `GET /conversations`: Lists all conversations belonging to the authenticated user.
  - `POST /conversations`: Creates a new conversation session.
  - `GET /conversations/{id}`: Loads full message history in chronological order.
  - `PATCH /conversations/{id}`: Updates chat title.
  - `DELETE /conversations/{id}`: Deletes a specific chat session.
  - `DELETE /conversations`: Clears all chat history for the user.
  - `POST /conversations/{id}/messages`: Appends user or assistant messages.

### `preferences.py`
* **Purpose:** User topic and interface preference endpoints.
* **Endpoints:** `GET /preferences`, `POST /preferences`.

### `recommendations.py`
* **Purpose:** Generates suggested research prompts based on user preferences and current knowledge base topics.
* **Endpoints:** `GET /recommendations`.

### `score_store.py`
* **Purpose:** Thread-safe reader and writer for historical peer-evaluation logs.
* **Key Functions:** `load_score_history()`, `append_score_entry()`
* **Target File:** `score_store.json`.

---

## 3. Persona Evolution Engine

Located in `persona_evolution/`, this subsystem allows personas to automatically adapt and evolve over time:

### `persona_evolution/tracker.py`
* **Purpose:** Computes rolling performance statistics for all personas across query history.
* **Key Functions:** `compute_rolling_averages()`, `load_score_history()`
* **Metrics:** Average received peer score, total queries evaluated, top-$K$ appearance rate.

### `persona_evolution/detector.py`
* **Purpose:** Detects underperforming personas that consistently score below quality thresholds.
* **Key Functions:** `detect_underperformers()`, `should_trigger_replacement()`
* **Thresholds:** Minimum evaluation queries before replacement (`MIN_QUERIES_BEFORE_REPLACE = 10`), score threshold (`REPLACEMENT_THRESHOLD = 7.0`).

### `persona_evolution/mutator.py`
* **Purpose:** Generates mutated replacement persona prompts.
* **Key Functions:** `generate_replacement_persona()`
* **What it does:** Synthesizes an upgraded persona prompt designed to fix the specific weaknesses (e.g. lack of depth, ungrounded claims) of the replaced persona.

### `persona_evolution/replacer.py`
* **Purpose:** Safely replaces an active persona in memory without restarting the server.
* **Key Functions:** `execute_replacement()`, `load_persona_versions()`
* **What it does:** Updates the active `PERSONAS` list, logs the event with timestamp and version number, and saves the history to `persona_evolution/versions.json`.

---

## 4. Frontend & Browser Interface

Located in `frontend/`, this provides a responsive single-page web interface:

### `frontend/index.html`
* **Purpose:** Main single-page web interface for EvoRAG.
* **Key Components:**
  - **Research Query Panel:** Textarea with keyboard shortcuts (`Ctrl+Enter`), personalized suggestion chips.
  - **Speech-to-Text (Voice Input):** Microphone button powered by Web Speech API supporting English, Kannada, Hindi, Tamil, and Telugu.
  - **Text-to-Speech (Voice Output):** Listen, pause, resume, and stop controls for reading consensus responses aloud.
  - **Consensus Panel:** Displays final synthesized answer and top persona badges.
  - **Persona Cards Grid:** Displays all 8 persona perspectives with expandable cards and score badges.
  - **Score Matrix:** Heatmap table showing persona-by-persona peer ratings across all dimensions.
  - **Evolution Chart:** Chart.js line graph plotting rolling persona score trends over query iterations.
  - **Multi-Turn Chat History Drawer:** User conversations sidebar with session switching, auto-naming, and deletion.
  - **Authentication Modals:** Seamless login/signup modals for user isolation.

### `frontend/auth.js`
* **Purpose:** Manages client-side JWT token storage (`localStorage`), user state, and authenticated HTTP headers.

### `frontend/styles.css`
* **Purpose:** Dark-theme CSS variables, card layouts, responsive media queries, score badges, and animations.

### `frontend/login.html`, `signup.html`, `onboarding.html`, `chat.html`
* **Purpose:** Standalone auxiliary pages for authentication, preference onboarding, and lightweight chat views.

---

## 5. Startup & Orchestration Scripts

### `run_scheduler.ps1`
* **Purpose:** PowerShell startup script for Step 1 (News Scheduler / Knowledge Base Ingestion).
* **Usage:**
  - `.\run_scheduler.ps1` (Default: checks index, runs ingestion if due, and schedules every 3 days)
  - `.\run_scheduler.ps1 -Once` (Runs check/ingestion once and exits)
  - `.\run_scheduler.ps1 -CheckOnly` (Reports whether an update is due)
  - `.\run_scheduler.ps1 -Force` (Forces an immediate fetch and incremental embedding run)

### `run_app.ps1`
* **Purpose:** PowerShell startup script for Step 2 (EvoRAG Application).
* **What it does:**
  - Verifies Python environment and FAISS index readiness.
  - Starts background Ollama process (`ollama serve`) if installed and not running.
  - Launches the default web browser to `http://localhost:8000/`.
  - Starts the FastAPI backend server on port 8000.

---

## 6. Testing & Validation Testbenches

### `test_phase6_orchestration.py`
* **Purpose:** Automated end-to-end test suite for Phase 6.
* **Tests:**
  1. Scheduler update-due logic and index reuse ($< 3$ days).
  2. Local retrieval sufficiency evaluation on in-domain vs. out-of-domain queries.
  3. Knowledge synthesis fallback generating all 8 personas without external API leakage.
  4. Dynamic index reloading upon disk file modification.

### `validate_phase1.py`
* **Purpose:** Validates news fetching, HTML stripping, URL deduplication, and chunk generation.

### `validate_phase2.py`
* **Purpose:** Validates MiniLM-L6-v2 embedding generation, FAISS index construction, disk persistence, and vector queries.

### `validate_phase3.py`
* **Purpose:** Validates FAISS candidate retrieval, citation prompt assembly, and single-turn RAG generation.

### `validate_phase4.py`
* **Purpose:** Validates concurrent execution of the 8 reasoning personas and empty-response retry logic.

### `validate_phase5.py`
* **Purpose:** Validates the $8 \times 8$ peer-evaluation scoring matrix, score normalization, top-3 persona selection, and consensus synthesis.

### `retrieval_diagnostics.py`
* **Purpose:** Measures retrieval diversity, URL distribution, and deduplication ratios across queries.

### `build_ai_corpus.py`
* **Purpose:** Utility script to run an initial sweep across AI topics (`OpenAI`, `DeepMind`, `Anthropic`, `AI regulation`) to build the initial knowledge corpus.

### `generate_framework_diagram.py`
* **Purpose:** Generates the official EvoRAG architecture diagram (`evorag_design_framework.png`) using matplotlib.

---

## 7. Data, Indexes & Storage

### `index/evorag.faiss`
* **Description:** Binary FAISS `IndexFlatL2` vector index containing dense 384-dimensional embeddings.

### `index/evorag_meta.json`
* **Description:** JSON sidecar holding metadata for every vector in the FAISS index (URL, title, source, publication date, chunk index, text).

### `index/ingestion_status.json`
* **Description:** Tracks the timestamp of the last successful knowledge update, total vector count, added chunks, and status.

### `evorag.db`
* **Description:** SQLite database storing registered users, chat sessions, message histories, and user preferences.

### `score_store.json`
* **Description:** Historical ledger storing detailed peer scores, winner selections, and synthesis details for every consensus query.

### `data/chunks_*.json`
* **Description:** Intermediate chunk files generated by `fetcher.py` containing processed news articles.

---

## 8. Documentation & Major Project Report

### `instruction.md`
* **Description:** Comprehensive implementation blueprint detailing project phases 1 through 6, constraints, architectural requirements, and scoring rubrics.

### `PROJECT_STATUS.md`
* **Description:** Summary of completed milestones, architecture components, and current operational status.

### `README.md`
* **Description:** High-level project summary, prerequisites, and quick-start instructions.

### `evorag_phase2/` (LaTeX Academic Report)
* **Description:** Complete academic documentation for the B.Tech Major Project submission.
  - `main_report.tex`: Root LaTeX project compiler.
  - `abstract.tex`, `introduction.tex`, `literature_review.tex`, `technical_description.tex`, `conclusion.tex`: Detailed project report chapters.
  - `certificate.tex`, `acknowledgement.tex`, `CourseOutcomes.tex`, `ProjectEvaluationRubrics.tex`: Formal academic declarations.
  - `*.png / *.jpeg`: High-Level Design (HLD), Low-Level Design (LLD), system architecture, and methodology flowcharts.

---

## Quick Reference Table

| File Name | Primary Function | Primary Phase |
| :--- | :--- | :--- |
| `pipeline.py` | Full multi-persona consensus orchestration & quality evaluation | Phase 5 / Phase 6 |
| `personas.py` | 8-Persona concurrent execution & prompt templates | Phase 4 |
| `voting.py` | Multi-dimensional peer scoring matrix construction | Phase 5 |
| `synthesis.py` | Top-K persona selection & consensus answer synthesis | Phase 5 |
| `retriever.py` | Semantic search, sufficiency check & dynamic index reloading | Phase 3 / Phase 6 |
| `embedder.py` | Sentence-Transformers embeddings & incremental FAISS updates | Phase 2 / Phase 6 |
| `fetcher.py` | Multi-source news fetching, cleaning & sliding-window chunking | Phase 1 |
| `knowledge_scheduler.py` | 3-Day automated news ingestion scheduler (APScheduler) | Phase 6 |
| `api.py` | FastAPI REST server hosting endpoints & web client | Phase 3–6 |
| `conversations.py` | Multi-turn chat persistence, naming & history management | Phase 6 |
| `auth.py` | JWT authentication & bcrypt password hashing | Phase 6 |
| `models.py` | SQLModel database table definitions | Phase 6 |
| `database.py` | SQLite database connection & session provider | Phase 6 |
| `run_scheduler.ps1` | PowerShell launcher for news ingestion scheduler | Phase 6 |
| `run_app.ps1` | PowerShell launcher for FastAPI & Web Browser | Phase 6 |
| `frontend/index.html` | Web UI with voice STT/TTS, persona cards, heatmap & chat | Phase 5 / Phase 6 |
| `test_phase6_orchestration.py` | Automated test suite for Phase 6 features | Phase 6 |
