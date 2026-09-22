"""
AgriRAG FastAPI Backend Server
Wraps the QueryGate pipeline from step6_query_gate.py with CORS-enabled REST endpoints.
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastapi.responses import FileResponse

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root and pipeline directory are in sys.path
WORKSPACE = Path(__file__).resolve().parent
PIPELINE_DIR = WORKSPACE / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from step6_query_gate import QueryGate

app = FastAPI(
    title="AgriRAG Advisory API",
    description="A Bias-Aware and Credibility-Driven Retrieval-Augmented Generation System for Agricultural Advisory",
    version="1.0.0"
)

# Enable CORS for browser frontend (supports file:// and localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global QueryGate instance
qg_instance = None


class QueryRequest(BaseModel):
    query: str


@app.on_event("startup")
def startup_event():
    global qg_instance
    print("🌾 Initializing AgriRAG QueryGate components...", flush=True)
    qg_instance = QueryGate()
    print("✅ AgriRAG QueryGate ready on http://localhost:8000!", flush=True)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "AgriRAG Advisory System",
        "endpoints": {
            "query": "POST /query",
            "docs": "GET /docs"
        }
    }


@app.get("/")
def serve_ui():
    ui_path = WORKSPACE / "ui" / "agrirag_ui.html"
    if not ui_path.exists():
        ui_path = WORKSPACE / "agrirag_ui.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return health()


@app.get("/hero_bg.jpg")
def serve_hero_image():
    img_path = WORKSPACE / "ui" / "hero_bg.jpg"
    if not img_path.exists():
        img_path = WORKSPACE / "hero_bg.jpg"
    if img_path.exists():
        return FileResponse(img_path)
    raise HTTPException(status_code=404, detail="Image not found")


@app.post("/query")
def process_query_endpoint(req: QueryRequest):
    global qg_instance
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if qg_instance is None:
        qg_instance = QueryGate()

    try:
        result = qg_instance.process_query(req.query.strip())
        return result
    except Exception as e:
        print(f"❌ Error processing query: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import time
    import threading
    import webbrowser
    import urllib.request
    import uvicorn

    def open_browser_when_ready():
        # Poll until server is actively responding
        for _ in range(60):
            time.sleep(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
        print("🚀 Server ready! Opening AgriRAG UI at http://localhost:8000 ...", flush=True)
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=open_browser_when_ready, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)

