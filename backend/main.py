import asyncio
import json
import os
from typing import AsyncGenerator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import (
    app as research_app,
    planner_app,
    searcher_app,
    build_writer_prompt,
    get_llm_stream,
)

fast = FastAPI(title="Multi-Agent Research System")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

fast.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class ResearchRequest(BaseModel):
    query: str
    depth: Optional[str] = "standard"


def _build_initial_state(request: ResearchRequest) -> dict:
    max_iter = {"fast": 1, "deep": 3}.get(request.depth, 2)
    return {
        "query":          request.query,
        "depth":          request.depth,
        "max_iterations": max_iter,
        "iteration":      0,
        "approved":       False,
        "sub_queries":    [],
        "search_results": [],
        "final_report":   "",
        "feedback":       "",
    }


# ---------- Blocking endpoint ----------
@fast.post("/research")
async def research(request: ResearchRequest):
    state = _build_initial_state(request)
    final_state = await asyncio.to_thread(research_app.invoke, state)
    return {"report": final_state["final_report"]}


# ---------- Streaming endpoint ----------
@fast.post("/research/stream")
async def research_stream(request: ResearchRequest):
    """
    SSE stream. Event shapes:
        {"type": "status",  "text": "...", "step": "planning|searching|writing|reviewing", "iteration": N}
        {"type": "token",   "text": "..."}
        {"type": "done",    "text": ""}
        {"type": "error",   "text": "..."}

    Flow:
        All iterations: plan → search → write (silent) → review
        After the loop exits (approved or max_iter hit): stream the
        final approved report to the client once.

    This means the user never sees a partial draft get wiped.
    They see status labels updating throughout, then the finished
    report streams in one clean pass at the end.

    Exception: fast mode skips review and streams the single draft live.
    """
    initial_state = _build_initial_state(request)

    async def event_generator() -> AsyncGenerator[str, None]:
        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"

        async def iter_writer_tokens(current_state: dict) -> AsyncGenerator[str, None]:
            """Async generator — yields LLM tokens one by one (for final streaming)."""
            prompt      = build_writer_prompt(current_state)
            llm         = get_llm_stream(request.depth)
            token_queue: asyncio.Queue = asyncio.Queue()

            def _run():
                for chunk in llm.stream(prompt):
                    t = chunk.content
                    if t:
                        token_queue.put_nowait(t)
                token_queue.put_nowait(None)

            task = asyncio.get_event_loop().run_in_executor(None, _run)
            while True:
                token = await token_queue.get()
                if token is None:
                    break
                yield token
            await task

        async def run_writer_silent(current_state: dict) -> str:
            """Run the writer and collect the full text without streaming tokens."""
            prompt      = build_writer_prompt(current_state)
            llm         = get_llm_stream(request.depth)
            token_queue: asyncio.Queue = asyncio.Queue()

            def _run():
                for chunk in llm.stream(prompt):
                    t = chunk.content
                    if t:
                        token_queue.put_nowait(t)
                token_queue.put_nowait(None)

            task = asyncio.get_event_loop().run_in_executor(None, _run)
            collected = []
            while True:
                token = await token_queue.get()
                if token is None:
                    break
                collected.append(token)
            await task
            return "".join(collected)

        try:
            from agent import reflector_node, revision_router

            current_state = dict(initial_state)
            max_iter      = current_state["max_iterations"]

            # ── Fast mode: single pass, stream live, no review ────────────
            if request.depth == "fast":
                yield sse({"type": "status", "text": "Planning research...",
                           "step": "planning", "iteration": 1})
                await asyncio.sleep(0)
                planned_state = await asyncio.to_thread(planner_app.invoke, current_state)

                if planned_state.get("approved") and planned_state.get("final_report"):
                    yield sse({"type": "status", "text": "Loaded from cache.",
                               "step": "writing", "iteration": 1})
                    for ch in planned_state["final_report"]:
                        yield sse({"type": "token", "text": ch})
                    yield sse({"type": "done", "text": ""})
                    return

                yield sse({"type": "status", "text": "Searching the web...",
                           "step": "searching", "iteration": 1})
                await asyncio.sleep(0)
                searched_state = await asyncio.to_thread(searcher_app.invoke, planned_state)

                yield sse({"type": "status", "text": "Writing report...",
                           "step": "writing", "iteration": 1})
                async for token in iter_writer_tokens(searched_state):
                    yield sse({"type": "token", "text": token})

                yield sse({"type": "done", "text": ""})
                return

            # ── Standard / Deep mode: all iterations silent, stream at end ─
            for iteration in range(1, max_iter + 1):
                is_revision = iteration > 1

                # Planning
                plan_text = (
                    f"Replanning based on feedback... (revision {iteration - 1})"
                    if is_revision else
                    "Planning research..."
                )
                yield sse({"type": "status", "text": plan_text,
                           "step": "planning", "iteration": iteration})
                await asyncio.sleep(0)

                planned_state = await asyncio.to_thread(planner_app.invoke, current_state)

                # Cache hit — stream it directly and exit
                if planned_state.get("approved") and planned_state.get("final_report"):
                    yield sse({"type": "status", "text": "Loaded from cache.",
                               "step": "writing", "iteration": iteration})
                    for ch in planned_state["final_report"]:
                        yield sse({"type": "token", "text": ch})
                    yield sse({"type": "done", "text": ""})
                    return

                # Searching
                yield sse({"type": "status", "text": "Searching the web...",
                           "step": "searching", "iteration": iteration})
                await asyncio.sleep(0)

                searched_state = await asyncio.to_thread(searcher_app.invoke, planned_state)

                # Writing (silent — building the draft for the reviewer)
                write_label = (
                    f"Writing revised draft {iteration}..."
                    if is_revision else
                    "Writing draft..."
                )
                yield sse({"type": "status", "text": write_label,
                           "step": "writing", "iteration": iteration})
                await asyncio.sleep(0)

                full_text     = await run_writer_silent(searched_state)
                current_state = {**searched_state, "final_report": full_text}

                # Reviewing
                yield sse({"type": "status", "text": "Reviewing report quality...",
                           "step": "reviewing", "iteration": iteration})
                await asyncio.sleep(0)

                current_state = await asyncio.to_thread(reflector_node, current_state)
                decision      = revision_router(current_state)

                if decision == "end":
                    break
                # else: feedback is set on current_state, loop continues

            # ── Stream the final approved report ──────────────────────────
            yield sse({"type": "status", "text": "Streaming final report...",
                       "step": "writing", "iteration": current_state.get("iteration", 1)})

            async for token in iter_writer_tokens(current_state):
                yield sse({"type": "token", "text": token})

            yield sse({"type": "done", "text": ""})

        except Exception as exc:
            import traceback
            traceback.print_exc()
            yield sse({"type": "error", "text": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@fast.get("/health")
async def health():
    return {"status": "ok"}