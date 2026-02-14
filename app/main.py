"""
Loqo AI Agent v3 — FastAPI Server

Endpoints:
  GET  /             → Frontend UI
  POST /generate     → Generate + self-correct image (REST)
  WS   /ws/generate  → Real-time generation with live updates
  GET  /health       → Health check
"""
import io
import json
import time
import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.agent import run_agent
from app.config import HOST, PORT, CORS_ORIGINS
from app.generator import describe_image

# ── Constants ───────────────────────────────────────────────────────
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n  Loqo AI Agent v3 — Self-Correcting Image Generator")
    print("  Gemini 2.5 Flash Image + LATS + Reflexion")
    print(f"  → http://localhost:{PORT}\n")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Loqo AI Agent v3",
    description="Self-correcting image generation with LATS tree search + Reflexion",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for CSS/JS assets
import os
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    from app.config import GEMINI_API_KEY
    return {
        "status": "ok",
        "version": "3.0.0",
        "gemini_configured": bool(GEMINI_API_KEY),
        "architecture": "Generate → Evaluate → Reflect → Rewrite Prompt → Regenerate",
        "research_basis": [
            "LATS (ICML 2024) — Tree Search over Prompt Strategies",
            "Reflexion (NeurIPS 2023) — Episodic Memory + Verbal RL",
            "DSPy Assertions (NeurIPS 2023) — Self-Verifying Evaluation",
            "Q-Align (ICML 2024) — Quality Level Mapping",
            "Gemini 2.5 Flash Image — Native Image Gen/Edit",
        ],
    }


@app.post("/generate")
async def generate_image(
    prompt: str = Form(""),
    criteria: str = Form(...),
    image: UploadFile = File(None),
):
    """Generate a self-correcting image from prompt + criteria."""
    start = time.time()
    
    criteria_list = [c.strip() for c in criteria.split(",") if c.strip()]
    if not criteria_list:
        return JSONResponse(status_code=400, content={"error": "No criteria provided."})
    
    # Parse uploaded image if present (with size guard)
    input_image = None
    if image and image.filename:
        from PIL import Image as PILImage
        import io
        contents = await image.read()
        if len(contents) > MAX_UPLOAD_SIZE:
            return JSONResponse(status_code=413, content={"error": f"Image too large. Max {MAX_UPLOAD_SIZE // (1024*1024)}MB."})
        input_image = PILImage.open(io.BytesIO(contents)).convert("RGB")
    
    # Auto-describe image if no prompt provided (criteria-aware)
    if not prompt and input_image is not None:
        prompt = await describe_image(input_image, criteria=criteria_list)
    elif not prompt:
        return JSONResponse(status_code=400, content={"error": "Provide either a prompt or an image."})
    
    result = await run_agent(prompt, criteria_list, input_image=input_image)
    result["processing_time_seconds"] = round(time.time() - start, 2)
    return _serialize(result)


@app.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket):
    """WebSocket endpoint for real-time generation updates."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        
        prompt = data.get("prompt", "")
        criteria_list = data.get("criteria", [])
        if isinstance(criteria_list, str):
            criteria_list = [c.strip() for c in criteria_list.split(",") if c.strip()]
        
        if not criteria_list:
            await websocket.send_json({"type": "error", "message": "Missing criteria"})
            return
        
        # Parse uploaded image if present (base64)
        input_image = None
        if data.get("image_b64"):
            import io
            from PIL import Image as PILImage
            img_bytes = base64.b64decode(data["image_b64"])
            input_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Auto-describe image if no prompt provided (criteria-aware)
        if not prompt and input_image is not None:
            await websocket.send_json({"type": "status", "message": "Analyzing uploaded image..."})
            prompt = await describe_image(input_image, criteria=criteria_list)
            await websocket.send_json({"type": "status", "message": f"Auto-prompt: {prompt[:120]}..."})
        elif not prompt:
            await websocket.send_json({"type": "error", "message": "Provide either a prompt or an image"})
            return
        
        async def on_iteration(iter_data):
            await websocket.send_json({
                "type": "iteration",
                "data": _serialize(iter_data),
            })
        
        start = time.time()
        result = await run_agent(prompt, criteria_list, on_iteration=on_iteration, input_image=input_image)
        result["processing_time_seconds"] = round(time.time() - start, 2)
        
        await websocket.send_json({"type": "result", "data": _serialize(result)})
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        import traceback
        print(f"[WS ERROR] {e}")
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def _serialize(obj):
    """Make objects JSON-safe (handles numpy types from evaluators)."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, Image.Image):
        return None  # Don't serialize PIL images directly
    # Handle numpy scalar types (from OpenCV/numpy evaluators)
    try:
        import numpy as np
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return round(float(obj), 4)
    except ImportError:
        pass
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
