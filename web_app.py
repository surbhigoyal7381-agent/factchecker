from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
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


app = FastAPI(title="Fake Checker", version="1.0.0")
app.add_middleware(
    @app.post("/api/image-check")
    async def image_check(file: UploadFile = File(...)) -> Dict[str, Any]:
        """Accept an image upload, save it, run ELA detector and optional reverse search.

        NOTE: Image validation (MIME/type/size/dim checks) has been removed per request.
        """
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")

        uploads_dir = STATIC_DIR / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename).name
        save_path = uploads_dir / safe_name
        try:
            contents = await file.read()
            with open(save_path, "wb") as fh:
                fh.write(contents)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

        try:
            from fact_checker import detect_image_fake
        except Exception:
            from .fact_checker import detect_image_fake

        result = detect_image_fake(str(save_path))
        file_url = f"/static/uploads/{safe_name}"

        # Attempt reverse image search if a BASE_URL and SERPAPI_API_KEY are available.
        reverse_matches: list[str] = []
        search_error: str | None = None
        base_url = os.getenv("BASE_URL", "").rstrip("/")
        image_url = f"{base_url}{file_url}" if base_url else None

        serpapi_key = os.getenv("SERPAPI_API_KEY")
        if image_url and serpapi_key:
            try:
                from serpapi import GoogleSearch

                params = {
                    "engine": "google_reverse_image",
                    "image_url": image_url,
                    "api_key": serpapi_key,
                }
                search = GoogleSearch(params)
                resp = search.get_dict()

                # Collect potential source URLs from common fields if available.
                candidates = []
                for key in ("image_results", "inline_images", "organic_results", "results"):
                    for item in resp.get(key, []) or []:
                        url = item.get("link") or item.get("source") or item.get("original") or item.get("url")
                        if url:
                            candidates.append(url)

                # Deduplicate and keep first 8 matches
                seen = set()
                for u in candidates:
                    if u not in seen:
                        seen.add(u)
                        reverse_matches.append(u)
                        if len(reverse_matches) >= 8:
                            break
            except Exception as exc:
                search_error = str(exc)

        # Combine ELA heuristic with reverse search to produce a final label
        ela_verdict = (result.get("verdict") or "UNCERTAIN").upper()
        final_label = "UNCERTAIN"
        reasoning = []

        if ela_verdict == "FAKE" and not reverse_matches:
            final_label = "DEEPFAKE"
            reasoning.append("ELA indicates strong localized editing and no matching originals were found online.")
        elif ela_verdict == "REAL" and reverse_matches:
            final_label = "ORIGINAL"
            reasoning.append("Low ELA signal and matching images found online; likely an original or widely distributed photo.")
        elif ela_verdict == "REAL" and not reverse_matches:
            final_label = "ORIGINAL"
            reasoning.append("Low ELA signal; no exact online matches located (could be original or not indexed).")
        elif ela_verdict == "UNCERTAIN" and reverse_matches:
            final_label = "ORIGINAL"
            reasoning.append("Moderate ELA signal but matching images found online; likely an original or lightly edited variant.")
        elif ela_verdict == "FAKE" and reverse_matches:
            final_label = "UNCERTAIN"
            reasoning.append("ELA suggests editing but similar images exist online; this may be an edited variant of a real photo.")
        else:
            final_label = "UNCERTAIN"
            reasoning.append("Insufficient or conflicting signals from ELA and reverse search.")

        return {
            "file_url": file_url,
            "ela": result,
            "reverse_matches": reverse_matches,
            "search_error": search_error,
            "final_verdict": final_label,
            "reasoning": " ".join(reasoning),
        }

    total = 0
    try:
        with open(save_path, "wb") as out_f:
            while True:
                chunk = await file.read(1024 * 64)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    out_f.close()
                    save_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail=f"File too large (limit {max_bytes} bytes)")
                out_f.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    # Basic image sanity checks
    try:
        from PIL import Image
        img = Image.open(save_path)
        img.verify()
        img.close()
        img = Image.open(save_path)
        width, height = img.size
        img.close()
        max_dim = int(os.getenv("MAX_IMAGE_DIM", "8000"))
        if width > max_dim or height > max_dim:
            save_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Image dimensions too large (max {max_dim}px)")
    except HTTPException:
        raise
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Uploaded file is not a valid image: {exc}") from exc

    try:
        # import locally to avoid circular import issues
        from fact_checker import detect_image_fake
    except Exception:
        from .fact_checker import detect_image_fake

    result = detect_image_fake(str(save_path))
    file_url = f"/static/uploads/{unique_name}"

    # Attempt reverse image search if a BASE_URL and SERPAPI_API_KEY are available.
    reverse_matches: list[str] = []
    search_error: str | None = None
    base_url = os.getenv("BASE_URL", "").rstrip("/")
    image_url = f"{base_url}{file_url}" if base_url else None

    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if image_url and serpapi_key:
        try:
            from serpapi import GoogleSearch

            params = {
                "engine": "google_reverse_image",
                "image_url": image_url,
                "api_key": serpapi_key,
            }
            search = GoogleSearch(params)
            resp = search.get_dict()

            # Collect potential source URLs from common fields if available.
            candidates = []
            for key in ("image_results", "inline_images", "organic_results", "results"):
                for item in resp.get(key, []) or []:
                    url = item.get("link") or item.get("source") or item.get("original") or item.get("url")
                    if url:
                        candidates.append(url)

            # Deduplicate and keep first 8 matches
            seen = set()
            for u in candidates:
                if u not in seen:
                    seen.add(u)
                    reverse_matches.append(u)
                    if len(reverse_matches) >= 8:
                        break
        except Exception as exc:
            search_error = str(exc)

    # Combine ELA heuristic with reverse search to produce a final label
    ela_verdict = (result.get("verdict") or "UNCERTAIN").upper()
    final_label = "UNCERTAIN"
    reasoning = []

    if ela_verdict == "FAKE" and not reverse_matches:
        final_label = "DEEPFAKE"
        reasoning.append("ELA indicates strong localized editing and no matching originals were found online.")
    elif ela_verdict == "REAL" and reverse_matches:
        final_label = "ORIGINAL"
        reasoning.append("Low ELA signal and matching images found online; likely an original or widely distributed photo.")
    elif ela_verdict == "REAL" and not reverse_matches:
        final_label = "ORIGINAL"
        reasoning.append("Low ELA signal; no exact online matches located (could be original or not indexed).")
    elif ela_verdict == "UNCERTAIN" and reverse_matches:
        final_label = "ORIGINAL"
        reasoning.append("Moderate ELA signal but matching images found online; likely an original or lightly edited variant.")
    elif ela_verdict == "FAKE" and reverse_matches:
        final_label = "UNCERTAIN"
        reasoning.append("ELA suggests editing but similar images exist online; this may be an edited variant of a real photo.")
    else:
        final_label = "UNCERTAIN"
        reasoning.append("Insufficient or conflicting signals from ELA and reverse search.")

    return {
        "file_url": file_url,
        "ela": result,
        "reverse_matches": reverse_matches,
        "search_error": search_error,
        "final_verdict": final_label,
        "reasoning": " ".join(reasoning),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=True)
