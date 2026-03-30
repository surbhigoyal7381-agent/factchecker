"""Fact-Checking AI Agent using LangGraph + LangChain.

Create a .env file in the same directory as this script and add keys like:

OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o-mini
SEARCH_PROVIDER=tavily
TAVILY_API_KEY=tvly-your-tavily-key
SERPAPI_API_KEY=your-serpapi-key

You can switch SEARCH_PROVIDER to "serpapi" if preferred.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TRUSTED_DOMAINS: List[str] = [
    "reuters.com",
    "apnews.com",
    "snopes.com",
]

ALLOWED_VERDICTS = ("TRUE", "FALSE", "MISLEADING", "UNVERIFIED")


class FactCheckState(TypedDict, total=False):
    user_input: str
    extracted_claim: str
    search_results: str
    verdict: Literal["TRUE", "FALSE", "MISLEADING", "UNVERIFIED"]
    explanation: str
    sources: List[str]
    source_map: Dict[int, str]
    key_facts: List[str]
    reasoning: str
    report_markdown: str


class ClaimExtraction(BaseModel):
    extracted_claim: str = Field(
        ...,
        description="Single, testable factual claim with emotional language removed.",
    )


class JudgeOutput(BaseModel):
    verdict: Literal["TRUE", "FALSE", "MISLEADING", "UNVERIFIED"]
    explanation: str = Field(..., description="Short summary of why this verdict was selected.")
    reasoning: str = Field(
        ...,
        description="2-5 sentences with only evidence-based reasoning from the provided sources.",
    )
    key_facts: List[str] = Field(
        default_factory=list,
        description=(
            "Bullet-style factual points. Each fact must end with citation markers like [1], [2]."
        ),
    )


def get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing OPENAI_API_KEY in environment. Add it to your .env file."
        )

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        timeout=45,
        max_retries=2,
    )


def build_guardrailed_query(claim: str) -> str:
    domain_clause = " OR ".join(f"site:{domain}" for domain in TRUSTED_DOMAINS)
    return f"({domain_clause}) \"{claim}\""


def search_with_tavily(query: str, max_results: int = 5) -> tuple[str, List[str], Dict[int, str]]:
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise RuntimeError(
            "tavily-python not installed. Add it via requirements.txt"
        ) from exc

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("Missing TAVILY_API_KEY in environment.")

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=False,
    )

    entries = response.get("results", [])
    summaries: List[str] = []
    sources: List[str] = []
    source_map: Dict[int, str] = {}

    for i, item in enumerate(entries, start=1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = item.get("content", "No summary available.")
        summaries.append(f"[{i}] {title}\nURL: {url}\nSummary: {content}")
        if url:
            sources.append(url)
            source_map[i] = url

    if not summaries:
        summaries.append("No relevant results were returned from Tavily.")

    return "\n\n".join(summaries), sources, source_map


def search_with_serpapi(query: str, max_results: int = 5) -> tuple[str, List[str], Dict[int, str]]:
    try:
        from serpapi import GoogleSearch
    except ImportError as exc:
        raise RuntimeError(
            "google-search-results not installed. Add it via requirements.txt"
        ) from exc

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("Missing SERPAPI_API_KEY in environment.")

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": max_results,
    }
    search = GoogleSearch(params)
    response = search.get_dict()

    entries = response.get("organic_results", [])
    summaries: List[str] = []
    sources: List[str] = []
    source_map: Dict[int, str] = {}

    for i, item in enumerate(entries[:max_results], start=1):
        title = item.get("title", "Untitled")
        url = item.get("link", "")
        snippet = item.get("snippet", "No summary available.")
        summaries.append(f"[{i}] {title}\nURL: {url}\nSummary: {snippet}")
        if url:
            sources.append(url)
            source_map[i] = url

    if not summaries:
        summaries.append("No relevant results were returned from SerpAPI.")

    return "\n\n".join(summaries), sources, source_map


def intake_node(state: FactCheckState) -> FactCheckState:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You extract one objective, testable factual claim from user text. "
                "Remove emotional framing, opinions, and rhetorical language.",
            ),
            ("human", "User text:\n{user_input}"),
        ]
    )
    chain = prompt | llm.with_structured_output(ClaimExtraction)
    result = chain.invoke({"user_input": state["user_input"]})
    logger.info("Extracted claim: %s", result.extracted_claim)
    return {"extracted_claim": result.extracted_claim}


def researcher_node(state: FactCheckState) -> FactCheckState:
    claim = state.get("extracted_claim", "").strip()
    if not claim:
        return {
            "search_results": "No claim available for research.",
            "sources": [],
            "source_map": {},
        }

    query = build_guardrailed_query(claim)
    provider = os.getenv("SEARCH_PROVIDER", "tavily").lower().strip()
    logger.info("Running guarded search with provider=%s", provider)

    try:
        if provider == "serpapi":
            search_text, sources, source_map = search_with_serpapi(query)
        else:
            search_text, sources, source_map = search_with_tavily(query)
    except Exception as exc:
        logger.exception("Search failed")
        search_text = (
            "Search step failed due to configuration/runtime error. "
            f"Details: {exc}"
        )
        sources = []
        source_map = {}

    return {
        "search_results": search_text,
        "sources": sources,
        "source_map": source_map,
    }


def judge_node(state: FactCheckState) -> FactCheckState:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an impartial fact-checking judge. "
                "You MUST use ONLY the provided search evidence. "
                "Do NOT use prior knowledge or internal training data. "
                "If evidence is weak, conflicting, or missing, choose UNVERIFIED. "
                "Allowed verdicts: TRUE, FALSE, MISLEADING, UNVERIFIED.",
            ),
            (
                "human",
                "Claim:\n{claim}\n\n"
                "Search Evidence (only source of truth):\n{search_results}\n\n"
                "Return: verdict, explanation, detailed reasoning, and key_facts. "
                "Each key fact must include citation markers that refer to source ids, "
                "for example [1] or [2]. If evidence is missing, return an empty key_facts list.",
            ),
        ]
    )

    chain = prompt | llm.with_structured_output(JudgeOutput)
    result = chain.invoke(
        {
            "claim": state.get("extracted_claim", ""),
            "search_results": state.get("search_results", ""),
        }
    )

    verdict = result.verdict if result.verdict in ALLOWED_VERDICTS else "UNVERIFIED"
    explanation = result.explanation.strip()
    reasoning = result.reasoning.strip()
    key_facts = [fact.strip() for fact in result.key_facts if fact.strip()]

    return {
        "verdict": verdict,
        "explanation": explanation,
        "reasoning": reasoning,
        "key_facts": key_facts,
    }


def reporter_node(state: FactCheckState) -> FactCheckState:
    sources = state.get("sources", [])
    sources_md = "\n".join(f"- {url}" for url in sources) if sources else "- No sources captured"
    key_facts = state.get("key_facts", [])
    key_facts_md = (
        "\n".join(f"- {fact}" for fact in key_facts)
        if key_facts
        else "- No evidence-backed facts were extracted"
    )

    report = (
        "# Fact-Check Report\n\n"
        f"**User Input**: {state.get('user_input', '')}\n\n"
        f"**Extracted Claim**: {state.get('extracted_claim', '')}\n\n"
        f"## Verdict: {state.get('verdict', 'UNVERIFIED')}\n\n"
        f"**Explanation**: {state.get('explanation', 'No explanation generated.')}\n\n"
        "## Reasoning\n"
        f"{state.get('reasoning', 'No reasoning generated.')}\n\n"
        "## Key Facts (with citations)\n"
        f"{key_facts_md}\n\n"
        "## Sources Used\n"
        f"{sources_md}\n"
    )

    return {"report_markdown": report}


def build_graph():
    workflow = StateGraph(FactCheckState)

    workflow.add_node("intake_node", intake_node)
    workflow.add_node("researcher_node", researcher_node)
    workflow.add_node("judge_node", judge_node)
    workflow.add_node("reporter_node", reporter_node)

    workflow.add_edge(START, "intake_node")
    workflow.add_edge("intake_node", "researcher_node")
    workflow.add_edge("researcher_node", "judge_node")
    workflow.add_edge("judge_node", "reporter_node")
    workflow.add_edge("reporter_node", END)

    return workflow.compile()


def run_fact_check(user_input: str) -> FactCheckState:
    app = build_graph()
    initial_state: FactCheckState = {"user_input": user_input}
    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    # Quick local test input. Replace this with any headline or article excerpt.
    sample_input = (
        "Breaking: Scientists proved that drinking coffee doubles lifespan overnight!"
    )

    final_state = run_fact_check(sample_input)

    print(final_state.get("report_markdown", "No report generated."))
