import os
import hashlib
import operator
from typing import List, Dict, TypedDict, Optional, Annotated
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from tools.web_search import search_web
import chromadb
from chromadb.utils import embedding_functions

# ---------- Configuration ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment")

llm_standard = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, openai_api_key=OPENAI_API_KEY)
llm_deep     = ChatOpenAI(model="gpt-4o-mini",   temperature=0.2, openai_api_key=OPENAI_API_KEY)

llm_standard_stream = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, openai_api_key=OPENAI_API_KEY, streaming=True)
llm_deep_stream     = ChatOpenAI(model="gpt-4o-mini",   temperature=0.2, openai_api_key=OPENAI_API_KEY, streaming=True)

def get_llm(depth: str):
    return llm_deep if depth == "deep" else llm_standard

def get_llm_stream(depth: str):
    return llm_deep_stream if depth == "deep" else llm_standard_stream

# ChromaDB cache
chroma_client = chromadb.PersistentClient(path="./research_cache")
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)
collection = chroma_client.get_or_create_collection(
    name="reports",
    embedding_function=embedding_fn
)

# ---------- State ----------
class ResearchState(TypedDict):
    query: str
    depth: str
    sub_queries: List[str]
    search_results: Annotated[List[Dict], operator.add]
    final_report: str
    iteration: int
    max_iterations: int
    approved: bool
    feedback: str

class SearcherState(TypedDict):
    sub_query: str
    depth: str

# ---------- Cache ----------
def cached_report(query: str) -> Optional[str]:
    results = collection.query(query_texts=[query], n_results=1)
    if (results['distances'] and
        len(results['distances'][0]) > 0 and
        results['distances'][0][0] < 0.25):
        return results['documents'][0][0]
    return None

def cache_store(query: str, report: str):
    doc_id = hashlib.md5(query.encode()).hexdigest()
    collection.upsert(
        documents=[report],
        metadatas=[{"query": query}],
        ids=[doc_id]
    )

# ---------- Nodes ----------
def planner_node(state: ResearchState):
    # Only check cache on the very first iteration
    if state.get("iteration", 0) == 0:
        cached = cached_report(state["query"])
        if cached:
            return {**state, "final_report": cached, "approved": True}

    if state.get("feedback"):
        prompt = f"""The previous report was not satisfactory. The reviewer said:
{state['feedback']}

Generate 3-5 new, more specific sub-questions that directly address the gaps mentioned above.
Focus on what information is missing or needs deeper investigation.
Original query: {state['query']}
Return only sub-questions, one per line."""
    else:
        num_q = 5 if state["depth"] == "deep" else 3
        prompt = f"""Split the user's research query into {num_q} specific sub-questions that can be answered by web search.
Return only the sub-questions, one per line.

Query: {state['query']}"""

    llm = get_llm(state["depth"])
    response = llm.invoke(prompt)
    sub_queries = [line.strip() for line in response.content.split("\n") if line.strip()][:5]

    return {
        **state,
        "sub_queries": sub_queries,
        "search_results": [],          # clear results so each iteration searches fresh
        "iteration": state.get("iteration", 0) + 1,
    }

def continue_to_searchers(state: ResearchState) -> List[Send]:
    if state.get("approved") and state.get("final_report"):
        return [Send("__end__", state)]
    return [
        Send("searcher", {"sub_query": sq, "depth": state["depth"]})
        for sq in state["sub_queries"]
    ]

def _fan_out_searchers(state: ResearchState) -> List[Send]:
    """Fan-out used by the standalone searcher_app graph."""
    if state.get("approved") and state.get("final_report"):
        return [Send("__end__", state)]
    return [
        Send("searcher", {"sub_query": sq, "depth": state["depth"]})
        for sq in state["sub_queries"]
    ]

def searcher_node(state: SearcherState) -> dict:
    sq = state["sub_query"]
    limit = 5 if state.get("depth") == "deep" else 3
    try:
        results = search_web(sq, limit=limit)
    except Exception as e:
        print(f"[searcher] Search failed for '{sq}': {e}")
        results = []
    return {"search_results": results}

def writer_node(state: ResearchState):
    if state.get("approved") and state.get("final_report"):
        return state

    max_results = 12 if state["depth"] == "deep" else 8
    context = "\n\n".join([
        f"**{r['title']}**\nURL: {r['url']}\nExcerpt: {r['snippet'][:800]}"
        for r in state["search_results"][:max_results]
    ])

    revision_prompt = ""
    if state.get("feedback"):
        revision_prompt = f"\n\nThe previous version of this report had this critique:\n{state['feedback']}\nPlease revise the report to specifically address those issues.\n"

    prompt = build_writer_prompt_from_parts(state["query"], state["depth"], context, revision_prompt)

    llm = get_llm_stream(state["depth"])
    chunks = []
    for chunk in llm.stream(prompt):
        token = chunk.content
        if token:
            chunks.append(token)

    return {**state, "final_report": "".join(chunks)}


def build_writer_prompt_from_parts(query: str, depth: str, context: str, revision_prompt: str = "") -> str:
    if depth == "deep":
        mode_instructions = """This is a DEEP RESEARCH report. The following sections are MANDATORY:

1. ## Overview
   Broad context and background on the topic.

2. ## Key Findings
   Specific, cited findings. Name researchers, institutions, or officials where possible.
   Do NOT use vague phrases like "experts say" — name them.

3. ## Controversies and Conflicting Viewpoints
   REQUIRED. Explicitly compare what different sources say about the same events or claims.

4. ## Limitations and Open Questions
   REQUIRED. Address gaps in available data, reliability concerns, what remains unknown.

5. ## Conclusion
   Synthesize — do not just summarize.

Additional requirements:
- Target ~1200 words minimum.
- At least 6 unique inline citations with working URLs.
- Academic, analytical tone throughout."""
    else:
        mode_instructions = """Provide a clear, factual summary (~800 words for standard, ~500 for fast).
Include an introduction, key findings, and a conclusion."""

    return f"""Write a well-cited research report based on the user's query and the search results below.

Query: {query}

Search results:
{context}
{revision_prompt}
Credibility rules:
- Official government reports, named scientists, peer-reviewed findings -> cite directly.
- Single anonymous witness claims, unverified memos -> prefix with "An unverified account claimed..."
- Never place both categories with equal weight.

Requirements:
- Use markdown headings, bullet points, and inline links for citations.
- {mode_instructions}
- Be specific, cite sources, and avoid repetition.

Report:"""


def build_writer_prompt(state: ResearchState) -> str:
    """Public interface used by main.py for streaming."""
    max_results = 12 if state["depth"] == "deep" else 8
    context = "\n\n".join([
        f"**{r['title']}**\nURL: {r['url']}\nExcerpt: {r['snippet'][:800]}"
        for r in state["search_results"][:max_results]
    ])
    revision_prompt = ""
    if state.get("feedback"):
        revision_prompt = f"\n\nThe previous version of this report had this critique:\n{state['feedback']}\nPlease revise the report to specifically address those issues.\n"
    return build_writer_prompt_from_parts(state["query"], state["depth"], context, revision_prompt)


def reflector_node(state: ResearchState):
    if state["depth"] == "fast":
        cache_store(state["query"], state["final_report"])
        return {**state, "approved": True, "feedback": ""}

    # Deep mode: force one revision on the first draft
    if state["depth"] == "deep" and state["iteration"] == 1:
        return {
            **state,
            "approved": False,
            "feedback": (
                "The first draft needs deeper coverage. Generate sub-questions that explore: "
                "conflicting viewpoints between sources, specific limitations or unknowns in the field, "
                "and any recent developments not yet covered."
            ),
        }

    deep_checklist = """
DEEP MODE MANDATORY CHECKLIST:
[ ] Does the report contain '## Controversies and Conflicting Viewpoints' with real source comparison?
[ ] Does the report contain '## Limitations and Open Questions' with specific gaps?
[ ] Are at least 6 sources cited with inline links?
[ ] Are named researchers, officials, or institutions cited?
[ ] Is the report ~1200 words or more?
[ ] Are fringe claims flagged as less credible than official/scientific findings?

If ANY box is unchecked, output REVISE.
""" if state["depth"] == "deep" else ""

    prompt = f"""You are a critical research reviewer. Evaluate the following report for:
- Relevance to the original query: {state['query']}
- Use of citations (are sources properly linked and named?)
- Clarity and structure
- Possible gaps or missing information
{deep_checklist}
If the report is fully satisfactory, answer with exactly "PASS".
Otherwise, output "REVISE: <specific, actionable suggestions referencing exactly what is missing>".

Report:
{state['final_report']}
"""
    llm = get_llm(state["depth"])
    response = llm.invoke(prompt)
    content = response.content.strip()

    if "PASS" in content:
        cache_store(state["query"], state["final_report"])
        return {**state, "approved": True, "feedback": ""}
    else:
        return {
            **state,
            "approved": False,
            "feedback": content.replace("REVISE:", "").strip(),
        }

def revision_router(state: ResearchState) -> str:
    if state["approved"]:
        return "end"
    if state.get("iteration", 0) >= state.get("max_iterations", 2):
        cache_store(state["query"], state["final_report"])
        return "end"
    return "planner"


# ---------- Graph 1a: planner_app ----------
# Runs only the planner node so main.py can show "Planning..." while it works.
planner_builder = StateGraph(ResearchState)
planner_builder.add_node("planner", planner_node)
planner_builder.add_edge(START, "planner")
planner_builder.add_edge("planner", END)
planner_app = planner_builder.compile()

# ---------- Graph 1b: searcher_app ----------
# Runs only the parallel searchers so main.py can show "Searching..." while they work.
searcher_builder = StateGraph(ResearchState)
searcher_builder.add_node("searcher", searcher_node)
searcher_builder.add_conditional_edges(START, _fan_out_searchers)
searcher_builder.add_edge("searcher", END)
searcher_app = searcher_builder.compile()

# ---------- Graph 2: full app (blocking /research endpoint) ----------
builder = StateGraph(ResearchState)
builder.add_node("planner", planner_node)
builder.add_node("searcher", searcher_node)
builder.add_node("writer", writer_node)
builder.add_node("reflector", reflector_node)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", continue_to_searchers)
builder.add_edge("searcher", "writer")
builder.add_edge("writer", "reflector")
builder.add_conditional_edges("reflector", revision_router, {
    "planner": "planner",
    "end": END,
})

app = builder.compile()