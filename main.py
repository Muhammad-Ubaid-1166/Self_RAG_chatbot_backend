# fastapi_self_rag.py
# Requires: pip install -U fastapi uvicorn langgraph langchain-openai langchain-community \
#           langchain-core langchain-text-splitters langchain-cohere langchain-tavily \
#           python-dotenv httpx supabase pypdf numpy requests
#
# Also requires supabase_pdf_cohere.py in the same folder — it provides
# build_retriever() (Supabase PDF loading + fast chunking + Cohere embeddings
# + the hang-proof CosineRetriever).
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

from supabase_pdf_cohere import build_retriever

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────
# All raw provider/tool errors go here only — never returned to the client.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("self_rag")

# ─── ENV Validation ───────────────────────────────────────────────────
def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Missing required environment variable: {name}. "
            f"Please set it in your .env file or environment."
        )
    return value

GROQ_API_KEY = _require_env("groq_api_key_3")
_require_env("TAVILY_API_KEY")  # read implicitly by TavilySearch; validated here for a clean startup failure

llm = ChatOpenAI(
    model=" openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

# ─── State ──────────────────────────────────────────────────────────────
class State(TypedDict):
    question: str
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    web_query: str
    retry_count: int

class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if external documents are needed to answer reliably, else False."
    )

decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether retrieval is needed.\n"
            "Return JSON that matches this schema:\n"
            "{{'should_retrieve': boolean}}\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts, citations, or info likely not in the model.\n"
            "- should_retrieve=False for general explanations, definitions, or reasoning that doesn't need sources.\n"
            "- If unsure, choose True."
        ),
        ("human", "Question: {question}"),
    ]
)

should_retrieve_llm = llm.with_structured_output(RetrieveDecision, method="function_calling")

def decide_retrieval(state: "State"):
    decision: RetrieveDecision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(question=state["question"])
    )
    return {"need_retrieval": decision.should_retrieve}

direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only your general knowledge.\n"
            "Do NOT assume access to external documents.\n"
            "If you are unsure or the answer requires specific sources, say:\n"
            "'I don't know based on my general knowledge.'"
        ),
        ("human", "{question}"),
    ]
)

def generate_direct(state: State):
    out = llm.invoke(
        direct_generation_prompt.format_messages(
            question=state["question"]
        )
    )
    return {
        "answer": out.content
    }

# `retriever` is lazy-initialized on first retrieval request.
# Python resolves this name at call time, not at def time, so it's safe
# for this closure to reference it before the module finishes loading.
retriever = None

def _init_retriever():
    global retriever
    if retriever is not None:
        return
    logger.info("[RETRIEVER] Lazy-initializing retriever...")
    try:
        retriever = build_retriever()
        if retriever is None:
            logger.warning("[RETRIEVER] No retriever built — no PDFs loaded.")
    except Exception as e:
        logger.error(f"[RETRIEVER] Failed to build retriever: {e}")
        retriever = None

def retrieve(state: State):
    _init_retriever()
    if retriever is None:
        return {"docs": []}
    return {"docs": retriever.invoke(state["question"])}

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True if the document helps answer the question, else False."
    )

is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance.\n"
            "Return JSON that matches this schema:\n"
            "{{'is_relevant': boolean}}\n\n"
            "A document is relevant if it contains information useful for answering the question."
        ),
        (
            "human",
            "Question:\n{question}\n\nDocument:\n{document}"
        ),
    ]
)

relevance_llm = llm.with_structured_output(RelevanceDecision, method="function_calling")

def is_relevant(state: State):
    relevant_docs: List[Document] = []
    for doc in state["docs"]:
        decision: RelevanceDecision = relevance_llm.invoke(
            is_relevant_prompt.format_messages(
                question=state["question"],
                document=doc.page_content
            )
        )
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}

rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a business RAG assistant.\n"
            "Answer the user's question using ONLY the provided context.\n"
            "If the context does not contain enough information, say:\n"
            "'No relevant document found.'\n"
            "Do not use outside knowledge.\n"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Context:\n{context}\n"
        ),
    ]
)

def generate_from_context(state: State):
    context = "\n\n---\n\n".join(
        [d.page_content for d in state.get("relevant_docs", [])]
    ).strip()
    if not context:
        return {"answer": "No relevant document found.", "context": ""}
    out = llm.invoke(
        rag_generation_prompt.format_messages(
            question=state["question"],
            context=context
        )
    )
    return {"answer": out.content, "context": context}

class WebQuery(BaseModel):
    query: str

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6–14 words).\n"
            "- If the question implies recency, add (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query",
        ),
        ("human", "Question: {question}"),
    ]
)

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery, method="function_calling")

def rewrite_query_node(state: State):
    out = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}

tavily = TavilySearch(max_results=5)

def web_search_node(state: State):
    q = state.get("web_query") or state["question"]
    data = tavily.invoke({"query": q})
    docs = []
    for r in data.get("results", []):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        docs.append(
            Document(
                page_content=text,
                metadata={"source": "web", "url": url, "title": title},
            )
        )
    return {"docs": docs, "retry_count": state.get("retry_count", 0) + 1}

def no_relevant_docs(state: State):
    return {"answer": "No relevant document found.", "context": ""}

def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
    if state["need_retrieval"]:
        return "retrieve"
    return "generate_direct"

def route_after_relevance(state: State) -> Literal["generate_from_context", "rewrite_query", "no_relevant_docs"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    if state.get("retry_count", 0) >= 2:
        return "no_relevant_docs"
    return "rewrite_query"

g = StateGraph(State)
g.add_node("decide_retrieval", decide_retrieval)
g.add_node("generate_direct", generate_direct)
g.add_node("retrieve", retrieve)
g.add_node("is_relevant", is_relevant)
g.add_node("generate_from_context", generate_from_context)
g.add_node("rewrite_query", rewrite_query_node)
g.add_node("web_search", web_search_node)
g.add_node("no_relevant_docs", no_relevant_docs)

g.add_edge(START, "decide_retrieval")
g.add_conditional_edges(
    "decide_retrieval",
    route_after_decide,
    {
        "generate_direct": "generate_direct",
        "retrieve": "retrieve",
    },
)
g.add_edge("generate_direct", END)
g.add_edge("retrieve", "is_relevant")
g.add_conditional_edges(
    "is_relevant",
    route_after_relevance,
    {
        "generate_from_context": "generate_from_context",
        "rewrite_query": "rewrite_query",
        "no_relevant_docs": "no_relevant_docs",
    },
)
g.add_edge("rewrite_query", "web_search")
g.add_edge("web_search", "is_relevant")  # 🔁 circle back
g.add_edge("no_relevant_docs", END)
g.add_edge("generate_from_context", END)

graph = g.compile()

# ─── FastAPI ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    logger.info("[STARTUP] Self-RAG API starting...")
    yield
    logger.info("[SHUTDOWN] Done.")

app = FastAPI(title="Self-RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

class DocOut(BaseModel):
    content: str
    metadata: dict

class AskResponse(BaseModel):
    answer: str
    context: str
    relevant_docs: List[DocOut]



@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    try:
        result = graph.invoke({
            "question": payload.question,
            "docs": [],
            "relevant_docs": [],
            "context": "",
            "answer": "",
            "retry_count": 0,
        })
    except Exception:
        # Full traceback goes to the server log only — the client never
        # sees raw provider/tool error text.
        logger.exception(f"Self-RAG run failed for question: {payload.question!r}")
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable. Please try again shortly.",
        )
    return AskResponse(
        answer=result.get("answer", ""),
        context=result.get("context", ""),
        relevant_docs=[
            DocOut(content=d.page_content, metadata=d.metadata)
            for d in result.get("relevant_docs", [])
        ],
    )

# Run with: uvicorn fastapi_self_rag:app --reload
