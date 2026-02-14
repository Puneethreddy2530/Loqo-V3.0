"""
Loqo AI Agent v3 — Configuration
Self-correcting IMAGE GENERATION agent.
Generate → Evaluate → Reflect → Rewrite Prompt → Regenerate
"""
import os
from dotenv import load_dotenv
load_dotenv()

# ── Gemini API ──────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_GEN_MODEL = "gemini-2.5-flash-image"   # Image generation + editing
GEMINI_EVAL_MODEL = "gemini-2.5-flash"         # Evaluation + reasoning

# ── Agent Loop ──────────────────────────────────────────────────────
MAX_ITERATIONS = 5           # Max generate→evaluate→fix cycles per attempt
MAX_RESTARTS = 2             # Fresh restarts if 5 iters don't solve it (total attempts = 1 + MAX_RESTARTS)
PASS_THRESHOLD = 0.75        # Criterion passes if score >= this
CONFIDENCE_THRESHOLD = 0.70  # Below this → uncertain
OVERALL_PASS_RATIO = 1.0     # All criteria must pass

# ── Early Stopping (SELF-REFINE heuristic) ──────────────────────────
EARLY_STOP_SCORE = 0.88      # Stop early if score > this and no criterion regressing
STAGNATION_THRESHOLD = 0.02  # Score improvement below this = stagnant
STAGNATION_PATIENCE = 2      # Stop after N consecutive stagnant iterations

# ── LATS Tree Search (over prompt strategies) ───────────────────────
MAX_DEPTH = MAX_ITERATIONS
BRANCH_FACTOR = 2
UCT_EXPLORATION = 1.414      # √2 from MCTS theory

# ── Reflexion Episodic Memory ───────────────────────────────────────
MAX_EPISODIC_MEMORY = 3      # Keep last N reflections

# ── Self-Consistency (for Gemini evaluation) ────────────────────────
CONSISTENCY_SAMPLES = 1      # Set to 3 for voting (costs 3x API calls)
CONSISTENCY_AGREEMENT = 0.66

# ── Q-Align Quality Levels (Wu et al., ICML 2024) ──────────────────
QUALITY_LEVELS = {
    "excellent": (0.90, 1.00),
    "good":      (0.75, 0.90),
    "fair":      (0.50, 0.75),
    "poor":      (0.25, 0.50),
    "bad":       (0.00, 0.25),
}

# ── Optional Local Models (for ground-truth evaluation) ─────────────
USE_CLIP = False             # Set True if you have GPU VRAM for CLIP
USE_YOLO = False             # Set True if you want object detection
CLIP_MODEL = "openai/clip-vit-base-patch32"
YOLO_MODEL = "yolov8n.pt"
DEVICE = "cuda"              # Auto-fallback to CPU

# ── Image Generation Settings ───────────────────────────────────────
IMAGE_SIZE = "1024x1024"     # Default output size
SAVE_ALL_ITERATIONS = True   # Save each generated image

# ── Progressive Image Resolution ────────────────────────────────────
# Early iterations use smaller images for speed, final uses full res
PROGRESSIVE_IMAGE_SIZES = {
    "early": "512x512",       # Iterations 1–3 (faster generation)
    "final": "1024x1024",    # Last iteration (full quality)
}
PROGRESSIVE_THRESHOLD = 3    # Switch to full res at this iteration

# ── Server ──────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8001
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")  # Comma-separated in production
