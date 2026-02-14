# Loqo AI Agent v3 — Self-Correcting Image Generator

A **research-grade, self-correcting image generation agent** that uses Gemini 2.5 Flash Image for generation and iteratively evaluates, reflects on, and rewrites prompts to fix quality issues — powered by three ICML/NeurIPS 2024 papers.

> **TL;DR** — You give it a prompt and criteria. It generates an image, evaluates it against every criterion, figures out what went wrong, rewrites the prompt, and regenerates. Repeat until all criteria pass or it exhausts attempts.

---

## Table of Contents

- [How It Works (Architecture)](#how-it-works-architecture)
- [Research Foundations](#research-foundations)
- [Evaluation System (Multi-Tier)](#evaluation-system-multi-tier)
- [Self-Correction Loop](#self-correction-loop)
- [Key Mechanisms](#key-mechanisms)
- [Frontend Features](#frontend-features)
- [File Structure](#file-structure)
- [Configuration Reference](#configuration-reference)
- [Setup & Running](#setup--running)
- [Deployment](#deployment)
- [API Reference](#api-reference)

---

## How It Works (Architecture)

```
┌──────────────────────────────────────────────────────────────┐
│                      USER INPUT                              │
│  Prompt: "A red logo with bold text LOQO"                    │
│  Criteria: ["red color", "bold text", "clean design"]        │
│  (Optional: upload reference image)                          │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌───────┐ │
│  │ GENERATE │───▶│ EVALUATE │───▶│ AGGREGATE │───▶│REFLECT│ │
│  │(Gemini   │    │(OpenCV + │    │(Score +   │    │+ Prompt│ │
│  │ 2.5 Flash│    │ Gemini   │    │ LATS      │    │Rewrite│ │
│  │ Image)   │    │ Vision)  │    │ Backprop) │    │       │ │
│  └──────────┘    └──────────┘    └───────────┘    └───┬───┘ │
│       ▲                                               │     │
│       └───────────────────────────────────────────────┘     │
│                     LOOP (up to 5 iterations)                │
│                                                              │
│  If still failing → RESTART with episodic memory (up to 3x) │
│  Final step → 1024px polish pass                             │
└──────────────────────────────────────────────────────────────┘
```

### Step-by-Step Flow

1. **GENERATE** — Gemini 2.5 Flash Image creates an image from the prompt (or edits the existing one)
2. **EVALUATE** — Each criterion is evaluated in parallel using OpenCV (deterministic) + Gemini Vision (semantic)
3. **AGGREGATE** — Scores are combined, backpropagated through the LATS tree, and the quality guard kicks in
4. **REFLECT** — Failed criteria are analyzed, and Gemini rewrites the generation prompt to fix specific issues
5. **Loop or Stop** — If all criteria pass (or early-stop/stagnation detected), stop. Otherwise, loop back to step 1 with the new prompt

---

## Research Foundations

This agent implements three published research papers:

### 1. LATS — Language Agent Tree Search (Zhou et al., ICML 2024)

**What it does:** Treats prompt engineering as a tree search problem.

- Each tree node represents a different generation prompt
- UCT (Upper Confidence Trees) formula balances exploitation (best-scoring prompts) vs exploration (untried strategies)
- Scores backpropagate up the tree to learn which prompt strategies work

**Where in code:** `agent.py` → `TreeNode` class, `uct_value()`, `backpropagate()`

### 2. Reflexion — Verbal Reinforcement Learning (Shinn et al., NeurIPS 2023)

**What it does:** Maintains episodic memory of what went wrong with past generations.

Instead of just tracking scores, the agent remembers:
- What the prompt was
- What failed ("text was cut off at edges")
- What to fix ("ensure 20px padding around all text")
- Lessons learned

This memory carries across restarts, so the agent never repeats the same mistake.

**Where in code:** `agent.py` → `EpisodicMemory` class, `reflect_and_rewrite_node()`

### 3. DSPy Assertions (Khattab et al., NeurIPS 2023)

**What it does:** Self-verifying evaluation prompts.

The Gemini evaluator must:
- Cite specific visual evidence (not vague claims)
- Provide issues if score < 0.5
- Give actionable fix suggestions (not "improve it")
- Have quality levels that match the numeric score

If the evaluator gives a low score but no issues, it auto-retries with stricter instructions.

**Where in code:** `evaluators.py` → `eval_gemini_vision()`, DSPy assertion rules in the prompt

---

## Evaluation System (Multi-Tier)

Every criterion goes through a **routing system** that picks the right evaluators:

### Tier 1: Programmatic Evaluators (OpenCV — Zero Hallucination)

These use computer vision algorithms for ground-truth measurements:

| Evaluator | Triggers On | Method | What It Measures |
|-----------|-------------|--------|------------------|
| **Sharpness** | "sharp", "blur", "quality", "crisp", "HD" | Laplacian variance + Tenengrad (Sobel gradients) | Edge clarity and focus |
| **Color Analysis** | "red", "blue", "warm", "burgundy", etc. | HSV color space masking | Percentage of pixels matching target color |
| **Noise Level** | "noise", "grain", "clean" | Immerkær 1996 method | Signal-to-noise ratio |
| **Brightness/Contrast** | "lighting", "bright", "dark", "contrast" | HSV V-channel statistics | Brightness (mean) and contrast (std dev) |

### Tier 2: Gemini Vision (Semantic Understanding)

For every criterion, Gemini 2.5 Flash also evaluates the image semantically. It returns:
- `score` (0.0 – 1.0)
- `visual_evidence` — what it actually sees in the image
- `issues_found` — specific problems
- `fix_suggestion` — actionable instruction for the generator

### Score Aggregation

Scores from all evaluators are **confidence-weighted**:

```
final_score = Σ(score_i × confidence_i) / Σ(confidence_i)
```

Programmatic evaluators have higher confidence (0.88–0.95) than Gemini (0.90) because they don't hallucinate.

### Quality Levels (Q-Align, Wu et al., ICML 2024)

Scores map to discrete levels:

| Level | Score Range |
|-------|------------|
| Excellent | 0.90 – 1.00 |
| Good | 0.75 – 0.90 |
| Fair | 0.50 – 0.75 |
| Poor | 0.25 – 0.50 |
| Bad | 0.00 – 0.25 |

A criterion **passes** when `score ≥ 0.75` (configurable via `PASS_THRESHOLD`).

---

## Self-Correction Loop

### Prompt Rewriting (The Core Innovation)

When criteria fail, the `reflect_and_rewrite_node` sends Gemini a structured request:

```
Here's the original request: "..."
Current prompt: "..."
These criteria FAILED: [...]
Fix suggestions from evaluators: [...]
Past attempts and lessons: [...]

→ Rewrite the prompt to fix ALL failures.
```

The rewritten prompt is specific. For example:
- ❌ Before: "A red logo"
- ✅ After: "A logo with burgundy red (#800020) background, bold white sans-serif text 'LOQO' centered with 20px padding, clean vector style, no gradients"

### Restart Mechanism

If 5 iterations don't solve all criteria:
1. The agent **restarts** with a fresh generation
2. But it keeps all episodic memory from previous attempts
3. It starts from the best prompt found so far
4. Up to **3 total attempts** (1 initial + 2 restarts)

### Quality Guard

If an edit makes the image **worse** (score drops):
- The agent reverts to the best image so far
- Sets a `score_dropped` flag
- Next iteration **regenerates from scratch** instead of editing the degraded image

### Early Stopping

The agent stops early when:
1. **Confidence-based**: Score > 0.88 AND all criteria near the pass threshold
2. **Stagnation**: Score improvement < 0.02 for 2 consecutive iterations

### Progressive Image Resolution

To balance speed and quality:

| Iterations 1–3 | Iteration 4–5 | Final Polish |
|:---:|:---:|:---:|
| 512×512 | 512×512 | 1024×1024 |
| Fast generation | Fast generation | Full quality |

After the loop finishes, a **final 1024px polish pass** regenerates the best image at full resolution.

---

## Key Mechanisms

### Image Upload Mode

When you upload a reference image:
1. The **prompt field disables** automatically
2. Gemini analyzes the uploaded image + your criteria to **auto-generate a prompt**
3. The auto-prompt is criteria-aware — if you say "anime style", it won't describe a "realistic photo"
4. The uploaded image is used as the starting point for the first iteration

### Retry Logic

Both `generate_image()` and `edit_image()` retry up to **3 times** with exponential backoff (2s → 4s) on network errors (`httpx.ReadError`, DNS failures, etc.).

### API Call Tracking

Each full iteration costs: `1 generation + N evaluations + 1 rewrite = N + 2` API calls.

For 3 criteria over 5 iterations: `5 × (3 + 2) = 25` API calls per attempt.

---

## Frontend Features

The frontend (`static/index.html`) is a single-page app with real-time WebSocket updates:

### Iteration Timeline
- Scrollable horizontal timeline showing every generated image
- Each card shows: iteration number, attempt number, resolution badge, score, per-criterion breakdown
- **Restart separators** between attempts
- **Polish cards** highlighted with gold border and sparkle badge

### Image Zoom / Lightbox
- **Click any image** (iteration or final) → fullscreen lightbox modal
- **Download PNG / JPG** buttons in the lightbox
- **Escape key** to close

### Download
- **Download buttons** under the final result image (PNG and JPG)
- Files named `loqo-output-<timestamp>.png/jpg`

### Progress Bar
- Real-time progress updates via WebSocket
- Shows current iteration, score, and resolution
- Status messages for auto-describe and other processing steps

---

## File Structure

```
loqo-agent-v3/
├── app/
│   ├── __init__.py
│   ├── config.py          # All configuration constants
│   ├── agent.py           # LATS + Reflexion + LangGraph orchestration
│   ├── evaluators.py      # Multi-tier evaluation (OpenCV + Gemini)
│   └── generator.py       # Image generation/editing + auto-describe
│
├── static/
│   └── index.html         # Frontend (HTML + CSS + JS, single file)
│
├── start.py               # Quick start script (env check + uvicorn)
├── requirements.txt       # Python dependencies
├── Procfile               # Heroku/Railway process file
├── railway.toml           # Railway deployment config
├── .env                   # Environment variables (GEMINI_API_KEY)
├── .env.example           # Template for .env
└── outputs/               # Saved iteration images (auto-created)
```

### Module Responsibilities

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `agent.py` | ~750 | LangGraph state machine, LATS tree, Reflexion memory, restart logic, quality guard, progressive sizing, final polish |
| `evaluators.py` | ~330 | Criterion routing, 4 programmatic evaluators, Gemini vision evaluator, DSPy-style assertions, confidence-weighted aggregation |
| `generator.py` | ~220 | Gemini image generation/editing with retry logic, edit-vs-regenerate decision engine, image utilities, auto-describe |
| `config.py` | ~70 | All tuneable parameters (thresholds, sizes, API models, server settings) |
| `main.py` | ~200 | FastAPI server, REST + WebSocket endpoints, CORS, upload guard, JSON serialization |

---

## Configuration Reference

All settings are in `app/config.py`:

### Agent Loop
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_ITERATIONS` | 5 | Generate→evaluate→fix cycles per attempt |
| `MAX_RESTARTS` | 2 | Fresh restarts if iterations don't solve it (total = 3 attempts) |
| `PASS_THRESHOLD` | 0.75 | Criterion passes if score ≥ this |
| `OVERALL_PASS_RATIO` | 1.0 | Fraction of criteria that must pass (1.0 = all) |

### Early Stopping
| Parameter | Default | Description |
|-----------|---------|-------------|
| `EARLY_STOP_SCORE` | 0.88 | Stop early if overall score > this |
| `STAGNATION_THRESHOLD` | 0.02 | Improvement below this = stagnant |
| `STAGNATION_PATIENCE` | 2 | Stop after N stagnant iterations |

### LATS Tree Search
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_DEPTH` | 5 | Tree depth (= MAX_ITERATIONS) |
| `BRANCH_FACTOR` | 2 | Children per node |
| `UCT_EXPLORATION` | 1.414 | √2 from MCTS theory |

### Image Generation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_SIZE` | 1024×1024 | Default output resolution |
| `PROGRESSIVE_IMAGE_SIZES` | 512 early, 1024 final | Resolution per stage |
| `PROGRESSIVE_THRESHOLD` | 3 | Switch to full res at this iteration |

### Models
| Parameter | Default | Description |
|-----------|---------|-------------|
| `GEMINI_GEN_MODEL` | gemini-2.5-flash-image | Image generation + editing |
| `GEMINI_EVAL_MODEL` | gemini-2.5-flash | Evaluation + prompt rewriting |

---

## Setup & Running

### Prerequisites
- Python 3.11+
- Gemini API key ([get one free](https://aistudio.google.com/apikey))

### Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Run
python start.py
```

Open **http://localhost:8001** in your browser.

### Usage Modes

**Mode 1 — Text-to-Image:**
1. Enter a prompt (e.g., "A professional product photo of a coffee mug")
2. Enter comma-separated criteria (e.g., "warm lighting, sharp details, wooden table")
3. Click **Generate & Self-Correct**

**Mode 2 — Image Upload (Reference-Based):**
1. Upload a reference image
2. The prompt field auto-disables
3. Enter criteria describing what you want (e.g., "anime style, vibrant colors, detailed background")
4. The agent auto-generates a criteria-aware prompt from your image
5. Click **Generate & Self-Correct**

---

## Deployment

### Railway

The project includes `railway.toml` and `Procfile` for one-click deployment:

```bash
# Set environment variables on Railway:
GEMINI_API_KEY=your_key
CORS_ORIGINS=https://yourdomain.com
PORT=8080  # Railway injects this
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key |
| `PORT` | ❌ | Server port (default: 8001) |
| `CORS_ORIGINS` | ❌ | Comma-separated allowed origins (default: `*`) |

---

## API Reference

### REST Endpoints

#### `GET /health`
Returns system status and configuration.

```json
{
  "status": "ok",
  "version": "3.0.0",
  "gemini_configured": true,
  "architecture": ["LATS", "Reflexion", "Q-Align", "Gemini 2.5 Flash Image"]
}
```

#### `POST /generate`
Generate an image with self-correction.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | No (if image provided) | Image generation prompt |
| `criteria` | string | Yes | Comma-separated evaluation criteria |
| `image` | file | No | Reference image upload |

### WebSocket Endpoint

#### `WS /ws/generate`
Real-time generation with live iteration updates.

**Send:**
```json
{
  "prompt": "A red logo",
  "criteria": ["red color", "bold text"],
  "image_b64": null
}
```

**Receive (iteration):**
```json
{
  "type": "iteration",
  "data": {
    "iteration": 1,
    "overall_score": 0.72,
    "all_passed": false,
    "criteria_results": [...],
    "image_b64": "base64...",
    "attempt": 1
  }
}
```

**Receive (status — during auto-describe):**
```json
{
  "type": "status",
  "message": "Analyzing uploaded image..."
}
```

**Receive (result):**
```json
{
  "type": "result",
  "data": {
    "status": "success",
    "final_score": 0.91,
    "best_image_b64": "base64...",
    "total_iterations": 4,
    "total_attempts": 1,
    "criteria_results": [...],
    "original_prompt": "...",
    "final_prompt": "...",
    "mistake_log": [...]
  }
}
```

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Image Generation | Gemini 2.5 Flash Image (native gen + edit) |
| Image Evaluation | Gemini 2.5 Flash + OpenCV |
| Agent Orchestration | LangGraph (StateGraph) |
| Server | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS with WebSocket |
| Deployment | Railway (Nixpacks) |

---

## License

MIT
