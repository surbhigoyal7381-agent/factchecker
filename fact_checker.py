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
import re
from typing import Dict, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from PIL import Image, ImageChops
from io import BytesIO

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GLOBAL_TRUSTED_DOMAINS: List[str] = [
    "reuters.com",
    "apnews.com",
    "snopes.com",
]

REGIONAL_TRUSTED_DOMAINS: Dict[str, List[str]] = {
    "IN": [
        "thehindu.com",
        "indianexpress.com",
        "hindustantimes.com",
        "timesofindia.indiatimes.com",
        "newindianexpress.com",
        "factly.in",
        "altnews.in",
    ],
    "US": [
        "npr.org",
        "nytimes.com",
        "washingtonpost.com",
        "wsj.com",
        "usatoday.com",
        "politifact.com",
    ],
    "UK": [
        "bbc.com",
        "bbc.co.uk",
        "theguardian.com",
        "ft.com",
        "independent.co.uk",
        "channel4.com",
        "fullfact.org",
    ],
    "CA": [
        "cbc.ca",
        "theglobeandmail.com",
        "thestar.com",
        "ctvnews.ca",
    ],
    "AU": [
        "abc.net.au",
        "smh.com.au",
        "theage.com.au",
        "theaustralian.com.au",
        "aap.com.au",
    ],
    "DE": [
        "dw.com",
        "spiegel.de",
        "faz.net",
    ],
    "FR": [
        "france24.com",
        "lemonde.fr",
        "liberation.fr",
    ],
}

ALLOWED_VERDICTS = ("TRUE", "FALSE", "MISLEADING", "UNVERIFIED")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "for",
    "from",
    "get",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "who",
    "with",
}

EXACTNESS_MARKERS = (
    "exactly",
    "precisely",
    "only",
    "just",
    "no more than",
)

COMPARATIVE_MARKERS = (
    "largest",
    "smallest",
    "biggest",
    "most",
    "least",
    "highest",
    "lowest",
    "worst",
    "best",
    "top",
    "number one",
    "ranked first",
    "more than",
    "less than",
    "higher than",
    "lower than",
    "ahead of",
    "behind",
)

EVIDENCE_COMPARATIVE_MARKERS = COMPARATIVE_MARKERS + (
    "ranked",
    "ranking",
    "index",
    "compared with",
    "compared to",
    "among",
)


class FactCheckState(TypedDict, total=False):
    user_input: str
    extracted_claim: str
    country_code: str
    region_reasoning: str
    trusted_domains: List[str]
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
    country_code: str = Field(
        default="GLOBAL",
        description=(
            "Most relevant country code for the claim context. Use ISO-like values such as IN, US, UK, CA, AU, DE, FR. "
            "Use GLOBAL if no country-specific context is identifiable."
        ),
    )
    region_reasoning: str = Field(
        default="No strong regional signal detected.",
        description="One short sentence explaining why this country/global context was selected.",
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


def _extract_number_and_unit(text: str) -> tuple[float | None, str | None]:
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(tonnes?|tons?|tonne|ton|kg|kilograms?|million|billion)?\b", text.lower())
    if not match:
        return None, None
    number = float(match.group(1))
    unit = match.group(2) if match.group(2) else None
    return number, unit


def _keyword_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return {tok for tok in tokens if tok not in STOPWORDS and len(tok) > 2}


def _parse_evidence_chunks(search_results: str) -> list[tuple[int, str]]:
    chunks: list[tuple[int, str]] = []
    for block in [b.strip() for b in search_results.split("\n\n") if b.strip()]:
        match = re.match(r"\[(\d+)\]\s", block)
        if not match:
            continue
        chunks.append((int(match.group(1)), block))
    return chunks


def has_comparative_semantics(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in COMPARATIVE_MARKERS)


def apply_comparative_logic(
    user_input: str,
    extracted_claim: str,
    search_results: str,
    current_verdict: str,
) -> tuple[str, str, str, list[str]] | None:
    """Require direct comparative evidence for superlative/ranking claims.

    Example: "India is the largest corrupted democracy" cannot be downgraded to
    "India has high corruption" and marked TRUE without evidence comparing India
    against other democracies.
    """
    original_has_comparative = has_comparative_semantics(user_input)
    extracted_has_comparative = has_comparative_semantics(extracted_claim)
    if not original_has_comparative and not extracted_has_comparative:
        return None

    evidence_chunks = _parse_evidence_chunks(search_results)
    claim_tokens = _keyword_tokens(user_input or extracted_claim)
    comparative_matches: list[tuple[int, str]] = []

    for idx, chunk in evidence_chunks:
        chunk_without_index = re.sub(r"^\[\d+\]\s*", "", chunk)
        overlap = len(claim_tokens.intersection(_keyword_tokens(chunk_without_index)))
        if overlap < 2:
            continue

        chunk_lower = chunk_without_index.lower()
        if any(marker in chunk_lower for marker in EVIDENCE_COMPARATIVE_MARKERS):
            comparative_matches.append((idx, chunk_without_index))

    # If the original claim was comparative/superlative but the extracted claim lost
    # that meaning and the evidence lacks direct comparative support, do not allow TRUE.
    if original_has_comparative and (not extracted_has_comparative or not comparative_matches):
        explanation = (
            "Comparative claim not fully supported: the original statement makes a ranking or superlative assertion, "
            "but the evidence only supports a weaker generic claim."
        )
        reasoning = (
            "The original input contains comparative or superlative language that requires direct evidence comparing India "
            "with other democracies or entities. Evidence about corruption levels alone does not establish that India is the "
            "largest, most, worst, or top case."
        )
        facts = [
            "Available evidence may show corruption concerns, but it does not directly prove the ranking-style wording in the original claim."
        ]

        if current_verdict == "TRUE":
            return "UNVERIFIED", explanation, reasoning, facts

    return None


def apply_numeric_logic(extracted_claim: str, search_results: str) -> tuple[str, str, list[str]] | None:
    """Apply conservative quantity logic when evidence clearly entails or contradicts a numeric claim."""
    claim_num, claim_unit = _extract_number_and_unit(extracted_claim)
    if claim_num is None:
        return None

    claim_tokens = _keyword_tokens(extracted_claim)
    if not claim_tokens:
        return None

    claim_lower = extracted_claim.lower()
    is_exact_claim = any(marker in claim_lower for marker in EXACTNESS_MARKERS)
    evidence_chunks = _parse_evidence_chunks(search_results)

    best_supporting: tuple[int, float, str] | None = None
    best_conflict: tuple[int, float, str] | None = None

    for idx, chunk in evidence_chunks:
        chunk_without_index = re.sub(r"^\[\d+\]\s*", "", chunk)
        ev_num, ev_unit = _extract_number_and_unit(chunk_without_index)
        if ev_num is None:
            continue

        # Reject clear unit mismatch when both sides specify units.
        if claim_unit and ev_unit and claim_unit[:3] != ev_unit[:3]:
            continue

        overlap = len(claim_tokens.intersection(_keyword_tokens(chunk_without_index)))
        if overlap < 2:
            continue

        if is_exact_claim:
            if ev_num == claim_num:
                best_supporting = (idx, ev_num, chunk)
            elif ev_num != claim_num:
                best_conflict = (idx, ev_num, chunk)
        else:
            if ev_num >= claim_num:
                if best_supporting is None or ev_num > best_supporting[1]:
                    best_supporting = (idx, ev_num, chunk)

    if is_exact_claim and best_conflict and not best_supporting:
        idx, ev_num, _ = best_conflict
        explanation = (
            "Numeric contradiction: claim asserts an exact quantity, but supporting evidence reports a different value."
        )
        reasoning = (
            f"The claim uses exact wording, requiring quantity {claim_num:g}. "
            f"Evidence reports {ev_num:g}, so the exact statement does not hold."
        )
        facts = [
            f"Evidence reports {ev_num:g} while the claim states exactly {claim_num:g}. [{idx}]"
        ]
        return "FALSE", explanation, facts + [reasoning]

    if best_supporting and not is_exact_claim:
        idx, ev_num, _ = best_supporting
        explanation = (
            "Mathematical entailment detected: evidence quantity is greater than or equal to the claimed quantity for the same event context."
        )
        reasoning = (
            f"The claim asserts at least {claim_num:g}. Evidence indicates {ev_num:g} for the same subject/context. "
            f"Since {claim_num:g} <= {ev_num:g}, the lower quantity is logically included."
        )
        facts = [
            f"Evidence cites {ev_num:g}, which includes the claimed {claim_num:g} as a subset quantity. [{idx}]"
        ]
        return "TRUE", explanation, facts + [reasoning]

    return None


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


def normalize_country_code(country_code: str) -> str:
    code = (country_code or "GLOBAL").strip().upper()
    if code in REGIONAL_TRUSTED_DOMAINS:
        return code
    return "GLOBAL"


def get_trusted_domains(country_code: str) -> List[str]:
    regional = REGIONAL_TRUSTED_DOMAINS.get(country_code, [])
    # Keep deterministic ordering and de-duplicate while preserving order.
    merged = GLOBAL_TRUSTED_DOMAINS + regional
    return list(dict.fromkeys(merged))


def build_guardrailed_query(claim: str, trusted_domains: List[str]) -> str:
    domain_clause = " OR ".join(f"site:{domain}" for domain in trusted_domains)
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
                "Remove emotional framing, opinions, and rhetorical language. "
                "Preserve all critical semantics from the original wording, especially negation, exact quantities, "
                "comparisons, rankings, superlatives, and time qualifiers. "
                "Do NOT weaken a strong claim into a broader or softer claim. "
                "Also classify the most relevant country context for this claim. "
                "If unclear, return GLOBAL.",
            ),
            ("human", "User text:\n{user_input}"),
        ]
    )
    chain = prompt | llm.with_structured_output(ClaimExtraction)
    result = chain.invoke({"user_input": state["user_input"]})
    country_code = normalize_country_code(result.country_code)
    logger.info("Extracted claim: %s", result.extracted_claim)
    logger.info("Detected region country code: %s", country_code)
    return {
        "extracted_claim": result.extracted_claim,
        "country_code": country_code,
        "region_reasoning": result.region_reasoning,
    }


def researcher_node(state: FactCheckState) -> FactCheckState:
    claim = state.get("extracted_claim", "").strip()
    country_code = normalize_country_code(state.get("country_code", "GLOBAL"))
    trusted_domains = get_trusted_domains(country_code)
    if not claim:
        return {
            "search_results": "No claim available for research.",
            "sources": [],
            "source_map": {},
            "trusted_domains": trusted_domains,
        }

    query = build_guardrailed_query(claim, trusted_domains)
    provider = os.getenv("SEARCH_PROVIDER", "tavily").lower().strip()
    logger.info("Running guarded search with provider=%s, country_code=%s", provider, country_code)

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
        "trusted_domains": trusted_domains,
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
                "Respect the original claim wording. Do not silently downgrade or broaden the claim. "
                "If the original claim is comparative, superlative, or ranking-based, require direct comparative evidence. "
                "Apply mathematical and logical consistency checks when numbers are present. "
                "If evidence says X and claim says a smaller non-exact quantity Y where Y <= X "
                "for the same event/entity, treat the claim as logically supported. "
                "If claim is exact and evidence differs, treat as not supported. "
                "If evidence is weak, conflicting, or missing, choose UNVERIFIED. "
                "Allowed verdicts: TRUE, FALSE, MISLEADING, UNVERIFIED.",
            ),
            (
                "human",
                "Claim:\n{claim}\n\n"
                "Original User Input:\n{user_input}\n\n"
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
            "user_input": state.get("user_input", ""),
            "search_results": state.get("search_results", ""),
        }
    )

    verdict = result.verdict if result.verdict in ALLOWED_VERDICTS else "UNVERIFIED"
    explanation = result.explanation.strip()
    reasoning = result.reasoning.strip()
    key_facts = [fact.strip() for fact in result.key_facts if fact.strip()]

    numeric_logic = apply_numeric_logic(
        state.get("extracted_claim", ""),
        state.get("search_results", ""),
    )
    if numeric_logic is not None:
        logic_verdict, logic_explanation, logic_facts = numeric_logic
        if logic_verdict == "TRUE" and verdict != "TRUE":
            verdict = "TRUE"
            explanation = logic_explanation
            reasoning = logic_facts[-1]
            key_facts = logic_facts[:-1]
        elif logic_verdict == "FALSE" and verdict == "TRUE":
            verdict = "FALSE"
            explanation = logic_explanation
            reasoning = logic_facts[-1]
            key_facts = logic_facts[:-1]

    comparative_logic = apply_comparative_logic(
        state.get("user_input", ""),
        state.get("extracted_claim", ""),
        state.get("search_results", ""),
        verdict,
    )
    if comparative_logic is not None:
        verdict, explanation, reasoning, key_facts = comparative_logic

    return {
        "verdict": verdict,
        "explanation": explanation,
        "reasoning": reasoning,
        "key_facts": key_facts,
    }


def detect_image_fake(image_path: str) -> dict:
    """
    Simple Error Level Analysis (ELA) based detector for potential image manipulation.

    Logic implemented:
    - Recompress the input image to JPEG at quality=90 and compute the absolute
      difference between the original and recompressed image (ELA image).
    - Convert ELA result to grayscale and compute the maximum and mean pixel
      differences as a lightweight signal of localized editing.
    - Heuristic thresholds:
        * If max_diff > 60 or mean_diff > 15 => verdict = FAKE
        * If max_diff > 30 or mean_diff > 6  => verdict = UNCERTAIN
        * Otherwise                              => verdict = REAL

    Notes:
    - This is a lightweight heuristic intended for a quick signal, not a
      definitive forensic tool. Use a dedicated model or service for production
      quality manipulation detection.

    Returns a dict with keys: verdict (REAL|FAKE|UNCERTAIN), score (mean), max, explanation
    """
    try:
        orig = Image.open(image_path).convert("RGB")
        buf = BytesIO()
        orig.save(buf, "JPEG", quality=90)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")
        ela = ImageChops.difference(orig, recompressed)
        ela_gray = ela.convert("L")
        extrema = ela_gray.getextrema() or (0, 0)
        max_diff = extrema[1]
        pixels = list(ela_gray.getdata())
        mean_diff = float(sum(pixels) / len(pixels)) if pixels else 0.0

        if max_diff > 60 or mean_diff > 15:
            verdict = "FAKE"
            explanation = "High error-level variations suggest possible manipulation or localized editing."
        elif max_diff > 30 or mean_diff > 6:
            verdict = "UNCERTAIN"
            explanation = "Moderate error-level variations; could be editing, heavy filtering, or recompression artifacts."
        else:
            verdict = "REAL"
            explanation = "Low error-level variation consistent with an unedited image or uniform compression."

        return {"verdict": verdict, "score": mean_diff, "max": max_diff, "explanation": explanation}
    except Exception as exc:
        return {"verdict": "UNCERTAIN", "score": 0.0, "max": 0, "explanation": f"Detection failed: {exc}"}


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
        f"**Detected Region**: {state.get('country_code', 'GLOBAL')}\n\n"
        f"**Region Logic**: {state.get('region_reasoning', 'No regional reasoning generated.')}\n\n"
        f"## Verdict: {state.get('verdict', 'UNVERIFIED')}\n\n"
        f"**Explanation**: {state.get('explanation', 'No explanation generated.')}\n\n"
        "## Reasoning\n"
        f"{state.get('reasoning', 'No reasoning generated.')}\n\n"
        "## Trusted Domains Used for Retrieval\n"
        + "\n".join(f"- {domain}" for domain in state.get("trusted_domains", GLOBAL_TRUSTED_DOMAINS))
        + "\n\n"
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
