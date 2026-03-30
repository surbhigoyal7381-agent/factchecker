from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fact_checker import run_fact_check

load_dotenv(override=True)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=3, max_length=5000)


class ChatResponse(BaseModel):
    verdict: Literal["TRUE", "FALSE", "MISLEADING", "UNVERIFIED"]
    extracted_claim: str
    country_code: str
    region_reasoning: str
    trusted_domains: List[str]
    explanation: str
    reasoning: str
    key_facts: List[str]
    sources: List[str]
    report_markdown: str


app = FastAPI(title="Fact-Checking AI Chat", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> Dict[str, Any]:
    text = payload.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = run_fact_check(text)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Fact-check pipeline failed: {exc}",
        ) from exc

    verdict = result.get("verdict", "UNVERIFIED")
    if verdict not in {"TRUE", "FALSE", "MISLEADING", "UNVERIFIED"}:
        verdict = "UNVERIFIED"

    return {
        "verdict": verdict,
        "extracted_claim": result.get("extracted_claim", ""),
        "country_code": result.get("country_code", "GLOBAL"),
        "region_reasoning": result.get("region_reasoning", "No regional reasoning available."),
        "trusted_domains": result.get("trusted_domains", []),
        "explanation": result.get("explanation", "No explanation available."),
        "reasoning": result.get("reasoning", "No reasoning available."),
        "key_facts": result.get("key_facts", []),
        "sources": result.get("sources", []),
        "report_markdown": result.get("report_markdown", ""),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=True)
