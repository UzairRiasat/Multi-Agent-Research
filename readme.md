# Multi-Agent Research & Report Generator

A full-stack AI research assistant that takes a natural language query, fans out parallel web searches, writes a structured report, and self-reviews it across multiple revision cycles — all streamed live to the browser.

Built with **LangGraph**, **FastAPI**, **Brave Search**, and **OpenAI**.

---

## Features

- **Three research depths** — Fast (single pass), Standard (one revision), Deep (up to three revisions with mandatory sections)
- **Parallel web search** — sub-queries run concurrently via LangGraph's fan-out pattern
- **Self-reviewing loop** — a Reflector node critiques each draft and sends targeted feedback back to the Planner
- **Live token streaming** — SSE stream delivers the final report word-by-word to the browser
- **Semantic cache** — ChromaDB stores approved reports; near-duplicate queries are served instantly without re-running the pipeline
- **Credibility rules** — the writer prompt distinguishes peer-reviewed / official sources from unverified claims
- **Copy & Download** — export the finished report as a `.md` file

---

## Architecture

```
User Query
    │
    ▼
┌─────────┐     cache hit → return immediately
│ Planner │ ──────────────────────────────────►  Final Report
└─────────┘
    │ sub-queries (3–5)
    ▼
┌──────────────────────────────────┐
│  Searcher × N  (parallel)        │  Brave Search API
└──────────────────────────────────┘
    │ search_results merged via reducer
    ▼
┌────────┐
│ Writer │  GPT-3.5-turbo (standard/fast) · GPT-4o-mini (deep)
└────────┘
    │ draft report
    ▼
┌───────────┐   PASS  ──► cache → stream to browser
│ Reflector │
└───────────┘   REVISE ──► feedback → Planner (next iteration)
```

The streaming endpoint splits the graph into three independently-invokable sub-graphs (`planner_app`, `searcher_app`, full `app`) so each pipeline stage can emit its own SSE status event while its work is actually running — Planning shows while the LLM generates sub-queries, Searching shows while HTTP requests are in flight.

### Nodes

| Node | Responsibility |
|---|---|
| **Planner** | Decomposes the query into sub-questions; incorporates Reflector feedback on revision passes; checks the semantic cache on the first iteration |
| **Searcher** | Runs one Brave Search request per sub-query; multiple instances execute in parallel |
| **Writer** | Synthesises search results into a structured markdown report; applies credibility rules; obeys depth-specific section requirements |
| **Reflector** | Scores the draft against a checklist; returns `PASS` (cache + stream) or `REVISE: <feedback>` (loop back) |

### Research Depths

| Mode | Model | Max iterations | Behaviour |
|---|---|---|---|
| Fast | gpt-3.5-turbo | 1 | Single pass, streamed live, no review |
| Standard | gpt-3.5-turbo | 2 | Silent draft → review → stream final |
| Deep | gpt-4o-mini | 3 | Forces revision on iteration 1; mandatory sections (Controversies, Limitations); ≥6 citations; ≥1200 words |

---

## Project Structure

```
multi-agent-research/
├── backend/
│   ├── agent.py          # LangGraph graphs, all node logic, cache
│   ├── main.py           # FastAPI app, SSE streaming endpoint
│   ├── tools/
│   │   └── web_search.py # Brave Search wrapper (sync + async)
│   ├── research_cache/   # ChromaDB persistent store (auto-created)
│   └── .env              # API keys (see below)
└── frontend/
    ├── index.html        # UI with pipeline tracker
    └── script.js         # SSE client, markdown rendering, scroll logic
```

---

## Setup

### Prerequisites

- Python 3.10+
- A [Brave Search API](https://brave.com/search/api/) key (free tier: 2 000 queries/month)
- An [OpenAI API](https://platform.openai.com/) key

### 1. Clone & install

```bash
git clone <your-repo-url>
cd multi-agent-research/backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install fastapi uvicorn langchain-openai langgraph chromadb python-dotenv httpx
```

### 2. Configure environment

Create `backend/.env`:

```env
OPENAI_API_KEY=sk-...
BRAVE_API_KEY=BSA...
# Optional: restrict CORS origins (defaults to localhost:3000)
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### 3. Run the backend

```bash
cd backend
uvicorn main:fast --reload --host 0.0.0.0 --port 8000
```

### 4. Open the frontend

Open `frontend/index.html` directly in your browser, or serve it with any static file server:

```bash
# Python built-in
cd frontend
python -m http.server 3000
```

Then visit `http://localhost:3000`.

---

## API Reference

### `POST /research`

Blocking endpoint. Returns the complete report when the full pipeline finishes.

```json
// Request
{ "query": "What are the latest trends in AI agents?", "depth": "standard" }

// Response
{ "report": "# Latest Trends...\n\n..." }
```

### `POST /research/stream`

SSE endpoint. Streams events as the pipeline runs.

```
data: {"type": "status",  "text": "Planning research...", "step": "planning",  "iteration": 1}
data: {"type": "status",  "text": "Searching the web...", "step": "searching", "iteration": 1}
data: {"type": "status",  "text": "Writing draft...",     "step": "writing",   "iteration": 1}
data: {"type": "status",  "text": "Reviewing...",         "step": "reviewing", "iteration": 1}
data: {"type": "token",   "text": "# Latest"}
data: {"type": "token",   "text": " Trends"}
...
data: {"type": "done",    "text": ""}
```

| Event type | When |
|---|---|
| `status` | Each pipeline stage starts |
| `token` | Each streamed word of the final report |
| `done` | Pipeline complete |
| `error` | Unhandled exception |

### `GET /health`

```json
{ "status": "ok" }
```

---

## How the Streaming Flow Works

For **fast** mode the report streams live as it is written. For **standard** and **deep** modes all revision drafts are written and reviewed silently (the pipeline tracker updates, but no text appears), and only the final approved report is streamed to the browser. This prevents the user from seeing a partial draft get wiped when a revision is triggered.

```
Fast:     Plan → Search → Stream live → Done

Standard/Deep (each iteration):
          Plan → Search → Write silent → Review
                                            │
                              REVISE ───────┘ (loop)
                              PASS  ──► Stream final report → Done
```

---

## Caching

Reports are stored in a local ChromaDB collection (`./research_cache`) using `text-embedding-3-small` embeddings. On the first iteration of any query, the planner checks for a semantically similar cached report (cosine distance < 0.25). A hit skips the entire pipeline and returns the cached report immediately.

Cache entries are written only after the Reflector approves a report, so partial or failed drafts are never cached.

---

## Customisation

| What | Where |
|---|---|
| Change models | `agent.py` — `llm_standard` / `llm_deep` |
| Adjust cache similarity threshold | `agent.py` — `cached_report()`, change `0.25` |
| Change max iterations per depth | `main.py` — `_build_initial_state()` dict |
| Add report sections | `agent.py` — `build_writer_prompt_from_parts()` |
| Change number of search results | `agent.py` — `searcher_node()` `limit` and `writer_node()` `max_results` |
| Adjust auto-scroll sensitivity | `script.js` — `userScrolledUp` threshold (`80` px) |

---

## Known Limitations

- The Brave Search free tier has a monthly query cap; deep mode uses up to 15 searches per run.
- ChromaDB runs in-process; for production use replace it with a hosted vector store.
- The blocking `/research` endpoint holds the connection for the full pipeline duration; prefer `/research/stream` for any user-facing use.