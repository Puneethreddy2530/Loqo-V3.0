"""
Loqo AI Agent v3 — Image Generator
Uses Gemini 2.5 Flash Image for generation and editing.

Two modes:
1. GENERATE: Text prompt → New image
2. EDIT: Existing image + correction prompt → Fixed image

The agent decides which mode based on evaluation results:
- Score < 0.4 → Regenerate from scratch (fundamentally wrong)
- Score >= 0.4 → Edit existing image (partially correct)
"""
import base64
import io
import os
import asyncio
from PIL import Image
from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY, GEMINI_GEN_MODEL, GEMINI_EVAL_MODEL

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, doubles each retry


# ── Initialize Client ───────────────────────────────────────────────

_client = None
_async_client = None

def get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client

def get_async_client():
    """Get the async client (client.aio) for true async API calls."""
    global _async_client
    if _async_client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env")
        _async_client = genai.Client(api_key=GEMINI_API_KEY).aio
    return _async_client


# ── Generate Image from Prompt ──────────────────────────────────────

async def generate_image(prompt: str, image_size: str = "1024x1024") -> Image.Image:
    """
    Generate a new image from a text prompt.
    Returns PIL Image. Retries on network errors.
    """
    client = get_async_client()
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.models.generate_content(
                model=GEMINI_GEN_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=types.ImageConfig(image_size=image_size),
                ),
            )
            
            if not response.candidates:
                raise RuntimeError("Gemini returned no candidates. Prompt may have been blocked.")
            
            parts = response.candidates[0].content.parts or []
            for part in parts:
                if part.inline_data is not None:
                    image_bytes = part.inline_data.data
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    return image
            
            raise RuntimeError("Gemini returned no image. Prompt may have been blocked.")
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"[GEN] Retry {attempt + 1}/{MAX_RETRIES} after {wait}s: {str(e)[:80]}")
                await asyncio.sleep(wait)
    
    raise RuntimeError(f"Generation failed after {MAX_RETRIES} retries: {last_error}")


# ── Edit Existing Image ─────────────────────────────────────────────

async def edit_image(image: Image.Image, correction_prompt: str) -> Image.Image:
    """
    Edit an existing image based on correction instructions.
    Retries on network errors.
    """
    client = get_async_client()
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.models.generate_content(
                model=GEMINI_GEN_MODEL,
                contents=[correction_prompt, image],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            
            if not response.candidates:
                raise RuntimeError("Gemini returned no candidates during edit.")
            
            parts = response.candidates[0].content.parts or []
            for part in parts:
                if part.inline_data is not None:
                    image_bytes = part.inline_data.data
                    edited = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    return edited
            
            raise RuntimeError("Gemini returned no image during edit.")
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"[EDIT] Retry {attempt + 1}/{MAX_RETRIES} after {wait}s: {str(e)[:80]}")
                await asyncio.sleep(wait)
    
    raise RuntimeError(f"Edit failed after {MAX_RETRIES} retries: {last_error}")


# ── Smart Generate/Edit Decision ────────────────────────────────────

async def generate_or_edit(
    original_prompt: str,
    current_image: Image.Image | None,
    improved_prompt: str,
    overall_score: float,
    failed_criteria: list[dict],
    image_size: str = "1024x1024",
    score_dropped: bool = False,
) -> Image.Image:
    """
    Decide whether to regenerate from scratch or edit the existing image.
    
    Decision logic:
    - No current image → Generate
    - Score < 0.4 → Regenerate (too broken to fix)
    - Score dropped from previous best → Regenerate (edit is degrading)
    - Score >= 0.4 → Edit (partially correct, fix specific issues)
    """
    if current_image is None:
        return await generate_image(improved_prompt, image_size=image_size)
    
    if overall_score < 0.4 or score_dropped:
        # Image is fundamentally wrong or degrading — regenerate from scratch
        return await generate_image(improved_prompt, image_size=image_size)
    
    # Image is partially correct — edit to fix specific issues
    issues = []
    for fc in failed_criteria:
        issues.append(f"- {fc.get('criterion', '')}: {fc.get('issue', 'needs improvement')}")
    
    correction = (
        f"Edit this image to fix the following issues:\n"
        + "\n".join(issues)
        + f"\n\nThe image should match this description: {improved_prompt}"
    )
    
    try:
        return await edit_image(current_image, correction)
    except Exception:
        return await generate_image(improved_prompt, image_size=image_size)


# ── Utility ─────────────────────────────────────────────────────────

def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def save_image(image: Image.Image, path: str):
    """Save PIL Image to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    image.save(path, "PNG")


# ── Auto-Describe Image ────────────────────────────────────────────

async def describe_image(image: Image.Image, criteria: list[str] = None) -> str:
    """
    Use Gemini to auto-generate an image generation prompt from an uploaded image.
    If criteria are provided, the prompt is tailored to the user's intent
    (e.g., "anime-style" criteria won't produce a literal photo description).
    """
    client = get_async_client()
    
    if criteria:
        criteria_text = ", ".join(criteria)
        instruction = (
            f"The user uploaded this image as a REFERENCE and wants to generate a new image "
            f"that meets these criteria: [{criteria_text}].\n\n"
            f"Write a detailed image generation prompt that:\n"
            f"1. Uses the uploaded image as inspiration/reference (describe key features: pose, expression, composition)\n"
            f"2. Targets the style and requirements from the criteria\n"
            f"3. Is written as a direct generation prompt, NOT a description of the original photo\n"
            f"4. Includes specific details: art style, colors, lighting, composition, mood\n\n"
            f"Output ONLY the prompt paragraph, nothing else."
        )
    else:
        instruction = (
            "Describe this image in detail for an image generation prompt. "
            "Include: subject matter, art style, colors, lighting, composition, "
            "mood, any visible text, and important visual elements. "
            "Write it as a single, detailed image generation prompt paragraph. "
            "Do NOT start with 'This image shows' — write it as a direct prompt."
        )
    
    response = await client.models.generate_content(
        model=GEMINI_EVAL_MODEL,
        contents=[instruction, image],
    )
    
    if not response.candidates:
        return "A detailed image"
    
    return response.text.strip()

