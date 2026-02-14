"""
Loqo AI Agent v3 — Evaluators

Evaluates generated images against user criteria.
Two tiers:
  1. Programmatic (OpenCV) — deterministic ground truth
  2. Gemini Vision — semantic understanding with DSPy-style assertions

Research basis:
- Q-Align (Wu et al., ICML 2024) — discrete quality levels
- DSPy Assertions (Khattab et al., NeurIPS 2023) — self-verifying prompts
"""
import json
import numpy as np
import cv2
from PIL import Image
from typing import Optional
from google import genai
from google.genai import types
from app.config import (
    GEMINI_API_KEY, GEMINI_EVAL_MODEL,
    QUALITY_LEVELS, PASS_THRESHOLD,
)

EvalResult = dict


# ═══════════════════════════════════════════════════════════════
# Q-Align Quality Level Mapping
# ═══════════════════════════════════════════════════════════════

def score_to_quality_level(score: float) -> str:
    for level, (low, high) in QUALITY_LEVELS.items():
        if low <= score <= high:
            return level
    return "unknown"


# ═══════════════════════════════════════════════════════════════
# TIER 1: Programmatic Evaluators (zero hallucination)
# ═══════════════════════════════════════════════════════════════

def eval_sharpness(image: Image.Image) -> EvalResult:
    """Laplacian + Tenengrad for robust sharpness detection."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.mean(gx**2 + gy**2)
    
    score = 0.5 * min(1.0, lap_var / 300) + 0.5 * min(1.0, tenengrad / 5000)
    return {
        "score": round(score, 3),
        "reasoning": f"Laplacian={lap_var:.0f}, Tenengrad={tenengrad:.0f}",
        "confidence": 0.95,
        "quality_level": score_to_quality_level(score),
    }


def eval_color_analysis(image: Image.Image, target_color: str) -> EvalResult:
    """HSV-based color presence detection."""
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    
    color_ranges = {
        "red":       [(0, 70, 50, 10, 255, 255), (170, 70, 50, 180, 255, 255)],
        "burgundy":  [(0, 70, 30, 10, 255, 120), (170, 70, 30, 180, 255, 120)],
        "orange":    [(10, 70, 50, 25, 255, 255)],
        "yellow":    [(25, 70, 50, 35, 255, 255)],
        "green":     [(35, 40, 40, 85, 255, 255)],
        "lightgreen":[(35, 20, 150, 85, 150, 255)],
        "blue":      [(85, 40, 40, 130, 255, 255)],
        "purple":    [(130, 40, 40, 170, 255, 255)],
        "pink":      [(140, 30, 100, 170, 255, 255)],
        "white":     [(0, 0, 200, 180, 30, 255)],
        "black":     [(0, 0, 0, 180, 255, 50)],
        "warm":      [(0, 50, 50, 30, 255, 255)],
        "cool":      [(85, 40, 40, 170, 255, 255)],
    }
    
    target = target_color.lower().strip().replace(" ", "")
    ranges = color_ranges.get(target)
    if not ranges:
        return {"score": 0.5, "reasoning": f"Unknown color '{target}'", "confidence": 0.2, "quality_level": "fair"}
    
    total = hsv.shape[0] * hsv.shape[1]
    matched = 0
    for r in ranges:
        mask = cv2.inRange(hsv, np.array(r[:3]), np.array(r[3:]))
        matched += cv2.countNonZero(mask)
    
    ratio = matched / total
    score = min(1.0, ratio * 3)
    return {
        "score": round(score, 3),
        "reasoning": f"'{target}' covers {ratio*100:.1f}% of pixels",
        "confidence": 0.92,
        "quality_level": score_to_quality_level(score),
    }


def eval_brightness_contrast(image: Image.Image) -> EvalResult:
    """Brightness + contrast analysis."""
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(float)
    brightness = v.mean()
    contrast = v.std()
    
    b_score = 1.0 if 80 <= brightness <= 200 else max(0.2, 1 - abs(brightness - 140) / 140)
    c_score = min(1.0, contrast / 60)
    score = 0.6 * b_score + 0.4 * c_score
    return {
        "score": round(score, 3),
        "reasoning": f"Brightness={brightness:.0f}/255, Contrast(std)={contrast:.0f}",
        "confidence": 0.90,
        "quality_level": score_to_quality_level(score),
    }


def eval_noise_level(image: Image.Image) -> EvalResult:
    """Noise estimation via Immerkær 1996 method."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    sigma = np.sum(np.abs(cv2.filter2D(gray, -1, M)))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    
    score = max(0.0, min(1.0, 1.0 - sigma / 30))
    return {
        "score": round(score, 3),
        "reasoning": f"Noise sigma={sigma:.2f} ({'clean' if sigma < 5 else 'noisy' if sigma > 15 else 'moderate'})",
        "confidence": 0.88,
        "quality_level": score_to_quality_level(score),
    }


# ═══════════════════════════════════════════════════════════════
# TIER 2: Gemini Vision Evaluator (semantic understanding)
# ═══════════════════════════════════════════════════════════════

_eval_client = None
_eval_async_client = None

def _get_eval_client():
    global _eval_client
    if _eval_client is None:
        _eval_client = genai.Client(api_key=GEMINI_API_KEY)
    return _eval_client

def _get_async_eval_client():
    """Get the async client (client.aio) for true parallel calls."""
    global _eval_async_client
    if _eval_async_client is None:
        _eval_async_client = genai.Client(api_key=GEMINI_API_KEY).aio
    return _eval_async_client


async def eval_gemini_vision(
    image: Image.Image,
    criterion: str,
    original_prompt: str = "",
    context: str = "",
    attempt: int = 0,
) -> EvalResult:
    """
    Gemini evaluation with DSPy Assertion pattern.
    Evaluates how well the generated image matches a specific criterion.
    
    Returns structured assessment with:
    - score (0-1)
    - specific visual evidence
    - actionable fix suggestions for the generator
    """
    if not GEMINI_API_KEY:
        return {"score": 0.5, "reasoning": "No API key", "confidence": 0.1, "quality_level": "fair"}
    
    context_section = f"CONTEXT (past failures):\n{context}" if context else ""
    prompt = f"""You are a STRICT image quality evaluator for AI-generated images.
The user asked for: "{original_prompt}"
Evaluate this generated image for ONE specific criterion.

CRITERION: "{criterion}"

{context_section}

RESPOND WITH EXACTLY THIS JSON (no markdown, no backticks, no extra text):
{{
    "quality_level": "<excellent|good|fair|poor|bad>",
    "score": <float 0.0 to 1.0>,
    "meets_criterion": <true|false>,
    "visual_evidence": "<SPECIFIC things you see — colors, objects, text, positions, sizes>",
    "issues_found": "<specific problems, or empty string if none>",
    "fix_suggestion": "<concrete instruction for the image generator to fix this — be specific about what to change>"
}}

RULES (DSPy Assertions):
1. visual_evidence MUST reference concrete, visible elements
2. If score < 0.5, issues_found MUST be non-empty
3. fix_suggestion must be actionable (e.g., "make the red darker" not "improve colors")
4. quality_level must match: excellent(0.9-1), good(0.75-0.9), fair(0.5-0.75), poor(0.25-0.5), bad(0-0.25)
5. Be STRICT. AI-generated images often have subtle issues — look carefully.
6. For text rendering: check if text is readable, correctly spelled, and properly placed."""

    try:
        async_client = _get_async_eval_client()
        response = await async_client.models.generate_content(
            model=GEMINI_EVAL_MODEL,
            contents=[prompt, image],
        )
        text = response.text.strip()
        
        # Clean markdown fences
        if "```" in text:
            text = text.split("```")[1] if text.count("```") >= 2 else text
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        
        parsed = json.loads(text)
        
        score = float(parsed.get("score", 0.5))
        level = parsed.get("quality_level", "fair")
        evidence = parsed.get("visual_evidence", "")
        issues = parsed.get("issues_found", "")
        fix = parsed.get("fix_suggestion", "")
        
        # DSPy Assertion: low scores must have issues
        if score < 0.5 and not issues and attempt < 1:
            return await eval_gemini_vision(
                image, criterion, original_prompt,
                context + "\nPREVIOUS: low score but no issues. Be specific.",
                attempt + 1,
            )
        
        # Auto-correct level if mismatched
        expected_level = score_to_quality_level(score)
        if level != expected_level:
            level = expected_level
        
        return {
            "score": score,
            "reasoning": f"[{level}] {evidence}. Issues: {issues or 'none'}",
            "confidence": 0.90 if parsed.get("meets_criterion") is not None else 0.5,
            "quality_level": level,
            "fix_suggestion": fix,
            "issues_found": issues,
        }
    except Exception as e:
        return {
            "score": 0.5,
            "reasoning": f"Eval error: {str(e)[:100]}",
            "confidence": 0.1,
            "quality_level": "fair",
            "fix_suggestion": "Unable to evaluate — retry",
            "issues_found": str(e)[:100],
        }


# ═══════════════════════════════════════════════════════════════
# Criterion Router — picks evaluators per criterion
# ═══════════════════════════════════════════════════════════════

COLOR_KW = {"color", "colour", "red", "blue", "green", "yellow", "orange", "purple",
            "pink", "white", "black", "brown", "warm", "cool", "burgundy", "light"}
QUALITY_KW = {"sharp", "blur", "focus", "quality", "resolution", "crisp", "clear",
              "hd", "detailed", "noise", "grain", "clean"}
LIGHT_KW = {"lighting", "light", "shadow", "bright", "dark", "exposure",
            "contrast", "backlit", "overexposed", "underexposed"}


async def evaluate_criterion(
    image: Image.Image,
    criterion: str,
    original_prompt: str = "",
    context: str = "",
) -> dict:
    """
    Multi-tier evaluation: programmatic checks + Gemini vision.
    Returns criterion result with fix suggestions for the generator.
    """
    words = set(criterion.lower().split())
    results = []
    
    # 1. Route to programmatic evaluators (ground truth)
    if words & COLOR_KW:
        color_words = words & COLOR_KW - {"color", "colour", "light", "dark"}
        for cw in color_words:
            results.append(("color_hsv", eval_color_analysis(image, cw)))
    
    if words & QUALITY_KW:
        results.append(("sharpness", eval_sharpness(image)))
        results.append(("noise", eval_noise_level(image)))
    
    if words & LIGHT_KW:
        results.append(("brightness", eval_brightness_contrast(image)))
    
    # 2. ALWAYS: Gemini vision evaluation (semantic understanding)
    if GEMINI_API_KEY:
        gemini_result = await eval_gemini_vision(
            image, criterion, original_prompt, context,
        )
        results.append(("gemini_vision", gemini_result))
    
    # ── Confidence-Weighted Aggregation ─────────────────────────
    if not results:
        return {
            "criterion": criterion, "score": 0.5, "confidence": 0.1,
            "details": [], "passed": False, "quality_level": "fair",
            "fix_suggestion": "No evaluators available",
        }
    
    total_conf = sum(r["confidence"] for _, r in results)
    weighted_score = sum(r["score"] * r["confidence"] for _, r in results) / total_conf
    avg_conf = total_conf / len(results)
    
    # Collect fix suggestions from evaluators
    fixes = [r.get("fix_suggestion", "") for _, r in results if r.get("fix_suggestion")]
    primary_fix = "; ".join(fixes[:3]) if fixes else ""
    
    # Collect issues
    all_issues = [r.get("issues_found", "") for _, r in results if r.get("issues_found")]
    
    return {
        "criterion": criterion,
        "score": round(weighted_score, 3),
        "confidence": round(avg_conf, 3),
        "passed": weighted_score >= PASS_THRESHOLD,
        "quality_level": score_to_quality_level(weighted_score),
        "details": [{"evaluator": name, **r} for name, r in results],
        "fix_suggestion": primary_fix,
        "issues_found": "; ".join(filter(None, all_issues)),
    }
