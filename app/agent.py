"""
Loqo AI Agent v3 — Self-Correcting Image GENERATION Agent

THE KEY DIFFERENCE FROM v2:
  v2: Evaluate same image 5x with different strategies (broken — score never improves)
  v3: Generate → Evaluate → Reflect → Rewrite Prompt → REGENERATE (score improves!)

RESEARCH BASIS:
1. LATS (Zhou et al., ICML 2024) — Tree search over PROMPT strategies
   - Each tree node = a different generation prompt
   - UCT balances exploitation (best prompts) vs exploration (novel prompts)
   - Backpropagate image scores to learn which prompt strategies work

2. Reflexion (Shinn et al., NeurIPS 2023) — Verbal reinforcement learning
   - Episodic memory stores what went wrong with each generated image
   - Prompt rewriting uses failure descriptions, not just scores
   - "The text was cut off at the edges" → "Ensure all text has 20px padding"

3. DSPy Assertions (Khattab et al., NeurIPS 2023)
   - Gemini evaluation prompts self-verify (evidence must be concrete)
   - Fix suggestions must be actionable for the generator

ARCHITECTURE:
  GENERATE (Gemini Image API) → EVALUATE (multi-tier) →
  AGGREGATE + BACKPROPAGATE → REFLECT (Reflexion) →
  REWRITE PROMPT (key step!) → back to GENERATE
"""
import asyncio
import json
import math
import time
import os
from typing import TypedDict, Any, Optional
from dataclasses import dataclass, field
from PIL import Image

from langgraph.graph import StateGraph, END

from app.evaluators import evaluate_criterion
from app.generator import generate_or_edit, image_to_base64, save_image
from app.config import (
    MAX_DEPTH, BRANCH_FACTOR, UCT_EXPLORATION,
    PASS_THRESHOLD, CONFIDENCE_THRESHOLD, MAX_EPISODIC_MEMORY,
    GEMINI_API_KEY, GEMINI_EVAL_MODEL, SAVE_ALL_ITERATIONS,
    MAX_RESTARTS, EARLY_STOP_SCORE, STAGNATION_THRESHOLD,
    STAGNATION_PATIENCE, PROGRESSIVE_IMAGE_SIZES, PROGRESSIVE_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════════
# LATS Tree Node (tracks prompt strategies and their image scores)
# ═══════════════════════════════════════════════════════════════

@dataclass
class TreeNode:
    """A node in the LATS prompt-search tree."""
    prompt: str                            # The generation prompt used
    score: float = 0.0                     # Cumulative score
    visits: int = 0                        # Times visited
    children: list = field(default_factory=list)
    parent: Optional[Any] = None
    image: Any = None                      # Generated PIL Image
    eval_results: list = field(default_factory=list)
    
    def uct_value(self, total_visits: int) -> float:
        """UCT formula from LATS paper."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.score / self.visits
        exploration = UCT_EXPLORATION * math.sqrt(
            math.log(total_visits) / self.visits
        )
        return exploitation + exploration
    
    def backpropagate(self, reward: float):
        """Propagate score up the tree."""
        node = self
        while node:
            node.visits += 1
            node.score += reward
            node = node.parent


# ═══════════════════════════════════════════════════════════════
# Reflexion Episodic Memory (Shinn et al., NeurIPS 2023)
# ═══════════════════════════════════════════════════════════════

@dataclass
class EpisodicMemory:
    """Stores structured reflections about generated images."""
    reflections: list = field(default_factory=list)
    prompts_tried: list = field(default_factory=list)
    best_score: float = 0.0
    best_results: list = field(default_factory=list)
    best_image: Any = None
    best_prompt: str = ""
    
    def add_reflection(self, reflection: dict):
        self.reflections.append(reflection)
        if len(self.reflections) > MAX_EPISODIC_MEMORY:
            self.reflections = self.reflections[-MAX_EPISODIC_MEMORY:]
    
    def add_prompt(self, prompt: str):
        self.prompts_tried.append(prompt)
    
    def get_context(self) -> str:
        """Build context for prompt rewriting from episodic memory."""
        if not self.reflections:
            return ""
        
        lines = ["PAST ATTEMPTS (learn from these):"]
        for i, ref in enumerate(self.reflections):
            lines.append(f"\n--- Attempt {ref.get('iteration', i+1)} ---")
            lines.append(f"Prompt used: {ref.get('prompt', '?')[:200]}")
            lines.append(f"Score: {ref.get('score', '?')}")
            lines.append(f"What failed: {ref.get('failures', 'N/A')}")
            lines.append(f"Fix suggestions: {ref.get('fixes', 'N/A')}")
            lines.append(f"Lesson: {ref.get('lesson', 'N/A')}")
        
        if self.prompts_tried:
            lines.append(f"\nPrompts already tried (DO NOT REPEAT):")
            for p in self.prompts_tried[-3:]:
                lines.append(f"  - {p[:150]}")
        
        return "\n".join(lines)
    
    def update_best(self, score: float, results: list, image: Any, prompt: str):
        if score > self.best_score:
            self.best_score = score
            self.best_results = results
            self.best_image = image
            self.best_prompt = prompt


# ═══════════════════════════════════════════════════════════════
# Agent State
# ═══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    original_prompt: str            # User's original generation prompt
    criteria: list[str]             # Evaluation criteria
    current_prompt: str             # Current (possibly rewritten) prompt
    current_image: Any              # Current generated PIL Image
    iteration: int
    max_iterations: int
    history: list[dict]             # Full iteration history
    memory: Any                     # EpisodicMemory
    tree_root: Any                  # TreeNode
    current_node: Any               # Current tree node
    current_results: list[dict]     # Current evaluation results
    done: bool
    images: list[dict]              # All generated images (base64 + prompt)
    api_calls: int                  # Total API calls made (cost tracking)


# ═══════════════════════════════════════════════════════════════
# Node: GENERATE (create or edit image)
# ═══════════════════════════════════════════════════════════════

async def generate_node(state: AgentState) -> dict:
    """
    Generate an image using the current prompt.
    First iteration: generate from scratch.
    Later iterations: edit or regenerate based on score.
    """
    current_prompt = state["current_prompt"]
    current_image = state.get("current_image")
    memory: EpisodicMemory = state["memory"]
    iteration = state["iteration"]
    
    # Determine failed criteria for edit mode
    failed_criteria = []
    if current_image is not None and state.get("current_results"):
        for r in state["current_results"]:
            if not r.get("passed", False):
                failed_criteria.append({
                    "criterion": r["criterion"],
                    "issue": r.get("issues_found", ""),
                    "fix": r.get("fix_suggestion", ""),
                })
    
    # Calculate overall score for edit vs regenerate decision
    overall_score = 0.0
    if state.get("current_results"):
        scores = [r["score"] for r in state["current_results"]]
        overall_score = sum(scores) / len(scores)
    
    # Determine image size: progressive sizing (512 early, 1024 final)
    is_final = iteration >= (state.get("max_iterations", MAX_DEPTH) - 1)
    image_size = PROGRESSIVE_IMAGE_SIZES["final"] if is_final else PROGRESSIVE_IMAGE_SIZES["early"]
    
    # Detect if quality guard reverted image (edit was degrading)
    score_dropped = False
    history = state.get("history", [])
    if history and history[-1].get("score_dropped", False):
        score_dropped = True
    
    # Generate or edit
    new_image = await generate_or_edit(
        original_prompt=state["original_prompt"],
        current_image=current_image,
        improved_prompt=current_prompt,
        overall_score=overall_score,
        failed_criteria=failed_criteria,
        image_size=image_size,
        score_dropped=score_dropped,
    )
    
    # Track in LATS tree
    tree_root: TreeNode = state["tree_root"]
    if iteration == 0:
        tree_root.prompt = current_prompt
        tree_root.image = new_image
        current_node = tree_root
    else:
        child = TreeNode(prompt=current_prompt, parent=tree_root, image=new_image)
        tree_root.children.append(child)
        current_node = child
    
    # Save iteration image
    images = state.get("images", [])
    img_record = {
        "iteration": iteration + 1,
        "prompt": current_prompt[:300],
        "image_b64": image_to_base64(new_image),
        "mode": "generate" if current_image is None or overall_score < 0.4 else "edit",
        "resolution": image_size,
    }
    images.append(img_record)
    
    if SAVE_ALL_ITERATIONS:
        os.makedirs("outputs", exist_ok=True)
        save_image(new_image, f"outputs/iter_{iteration + 1}.png")
    
    return {
        "current_image": new_image,
        "current_node": current_node,
        "images": images,
    }


# ═══════════════════════════════════════════════════════════════
# Node: EVALUATE (check generated image against criteria)
# ═══════════════════════════════════════════════════════════════

async def evaluate_node(state: AgentState) -> dict:
    """
    Evaluate the generated image against all criteria.
    PARALLEL: All criteria evaluated simultaneously via asyncio.gather.
    """
    image = state["current_image"]
    criteria = state["criteria"]
    original_prompt = state["original_prompt"]
    memory: EpisodicMemory = state["memory"]
    context = memory.get_context()
    
    # Run ALL criteria evaluations in parallel (not one-by-one)
    tasks = [
        evaluate_criterion(
            image, criterion,
            original_prompt=original_prompt,
            context=context,
        )
        for criterion in criteria
    ]
    results = await asyncio.gather(*tasks)
    
    return {
        "current_results": list(results),
        "iteration": state["iteration"] + 1,
    }


# ═══════════════════════════════════════════════════════════════
# Node: AGGREGATE + BACKPROPAGATE
# ═══════════════════════════════════════════════════════════════

def aggregate_node(state: AgentState) -> dict:
    """Score results, backpropagate through LATS tree, track best.
    
    QUALITY GUARD: If this iteration scored worse than the best so far,
    revert current_image to the best image. This ensures the next
    generate_node edits from the highest-quality base, never a degraded one.
    """
    results = state["current_results"]
    iteration = state["iteration"]
    memory: EpisodicMemory = state["memory"]
    current_node: TreeNode = state.get("current_node")
    
    overall_score = sum(r["score"] for r in results) / len(results)
    all_passed = all(r.get("passed", False) for r in results)
    
    # LATS backpropagation
    if current_node:
        current_node.eval_results = results
        current_node.backpropagate(overall_score)
    
    # Reflexion: track best (only updates if score > best_score)
    memory.update_best(overall_score, results, state["current_image"], state["current_prompt"])
    
    # ── QUALITY GUARD ──────────────────────────────────────────
    # If this iteration scored WORSE than the best, revert to the
    # best image so the next edit doesn't start from a degraded base.
    use_image = state["current_image"]
    score_dropped = False
    if memory.best_image is not None and overall_score < memory.best_score:
        use_image = memory.best_image
        score_dropped = True
    
    record = {
        "iteration": iteration,
        "criteria_results": results,
        "overall_score": round(overall_score, 3),
        "all_passed": all_passed,
        "prompt_used": state["current_prompt"][:200],
        "image_b64": image_to_base64(state["current_image"]) if state["current_image"] else None,
        "score_dropped": score_dropped,
        "best_score_so_far": round(memory.best_score, 3),
    }
    
    history = state.get("history", []) + [record]
    
    # ── CONFIDENCE-BASED EARLY STOP (SELF-REFINE heuristic) ────
    # Stop early if score is high enough, no criterion is far below threshold,
    # and no criterion is regressing vs previous iteration
    early_stop = False
    if overall_score >= EARLY_STOP_SCORE and not all_passed:
        # Ensure no single criterion is far below pass threshold
        all_near_pass = all(r["score"] >= PASS_THRESHOLD - 0.05 for r in results)
        if all_near_pass:
            # Check no criterion is regressing vs previous iteration
            prev_results = state.get("current_results", [])
            regressing = False
            if prev_results and len(prev_results) == len(results):
                for prev, curr in zip(prev_results, results):
                    if curr["score"] < prev.get("score", 0) - 0.05:
                        regressing = True
                        break
            if not regressing:
                early_stop = True
                all_passed = True  # Treat as passed — close enough
    
    # ── STAGNATION DETECTION (IoRT) ────────────────────────────
    # Stop if marginal improvement < threshold for N iterations
    # Use abs() so regression isn't silently treated as stagnation
    stagnant = False
    if len(history) >= STAGNATION_PATIENCE + 1:
        recent_scores = [h["overall_score"] for h in history[-(STAGNATION_PATIENCE + 1):]]
        improvements = [recent_scores[i+1] - recent_scores[i] for i in range(len(recent_scores) - 1)]
        if all(abs(imp) < STAGNATION_THRESHOLD for imp in improvements):
            stagnant = True
            all_passed = True  # No point continuing — diminishing returns
    
    # Track API calls accurately
    # Per full iteration: 1 gen + N eval + 1 rewrite = 2 + len(criteria)
    api_calls = state.get("api_calls", 0) + 2 + len(results)  # gen + eval + rewrite
    
    return {
        "history": history,
        "memory": memory,
        "done": all_passed,
        "current_image": use_image,  # Quality guard: always edit from best
        "api_calls": api_calls,
    }


# ═══════════════════════════════════════════════════════════════
# Node: REFLECT + REWRITE PROMPT (the key innovation)
# ═══════════════════════════════════════════════════════════════

async def reflect_and_rewrite_node(state: AgentState) -> dict:
    """
    Reflexion + Prompt Rewriting:
    1. Analyze what went wrong with the generated image
    2. Build structured reflection with lessons learned
    3. REWRITE the generation prompt to fix failures
    
    This is where the self-correction actually happens.
    """
    if state.get("done"):
        return {}
    
    results = state["current_results"]
    iteration = state["iteration"]
    memory: EpisodicMemory = state["memory"]
    
    failed = [r for r in results if not r.get("passed", False)]
    
    # Build structured reflection
    failures_desc = []
    fixes_desc = []
    for f in failed:
        failures_desc.append(f"{f['criterion']} (score: {f['score']:.2f})")
        if f.get("fix_suggestion"):
            fixes_desc.append(f"{f['criterion']}: {f['fix_suggestion']}")
        if f.get("issues_found"):
            fixes_desc.append(f"Issue: {f['issues_found']}")
    
    reflection = {
        "iteration": iteration,
        "prompt": state["current_prompt"][:200],
        "score": round(sum(r["score"] for r in results) / len(results), 3),
        "failures": failures_desc,
        "fixes": fixes_desc,
        "lesson": "",
    }
    
    # ── REWRITE THE PROMPT ──────────────────────────────────────
    # This is the core self-correction mechanism
    new_prompt = state["current_prompt"]
    
    if GEMINI_API_KEY and failed:
        try:
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            async_models = client.aio.models
            
            rewrite_prompt = f"""You are an expert prompt engineer for AI image generation.

ORIGINAL USER REQUEST: "{state['original_prompt']}"

CURRENT GENERATION PROMPT: "{state['current_prompt']}"

The generated image was evaluated and these criteria FAILED:
{json.dumps(failures_desc, indent=2)}

Specific fix suggestions from evaluators:
{json.dumps(fixes_desc, indent=2)}

{memory.get_context()}

TASK: Rewrite the generation prompt to fix ALL failed criteria.

RULES:
1. Keep everything that's working — only fix what failed
2. Be EXTREMELY specific about visual details (exact colors, sizes, positions)
3. If text rendering failed, specify: font size, placement, padding, contrast
4. If colors are wrong, use specific color names (e.g., "burgundy red #800020" not just "dark red")
5. Add explicit constraints like "ensure text is fully inside boxes with 10px padding"
6. The prompt should be a direct image generation instruction, not a conversation

Output ONLY the rewritten prompt, nothing else. No explanation, no markdown."""

            response = await async_models.generate_content(
                model=GEMINI_EVAL_MODEL,
                contents=[rewrite_prompt],
            )
            new_prompt = response.text.strip()
            
            # Clean up any markdown artifacts
            if new_prompt.startswith('"') and new_prompt.endswith('"'):
                new_prompt = new_prompt[1:-1]
            if new_prompt.startswith("```"):
                lines = new_prompt.split("\n")
                new_prompt = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            
            reflection["lesson"] = f"Rewrote prompt. Key changes: {new_prompt[:150]}"
            
        except Exception as e:
            # Fallback: append fix suggestions to existing prompt
            fix_additions = ". ".join(f.get("fix_suggestion", "") for f in failed if f.get("fix_suggestion"))
            if fix_additions:
                new_prompt = f"{state['current_prompt']}. IMPORTANT: {fix_additions}"
            reflection["lesson"] = f"Prompt rewrite failed ({str(e)[:50]}), appended fixes"
    else:
        reflection["lesson"] = "No API key or all criteria passed"
    
    memory.add_reflection(reflection)
    memory.add_prompt(new_prompt)
    
    return {
        "memory": memory,
        "current_prompt": new_prompt,
    }


# ═══════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> str:
    if state.get("done"):
        return "done"
    if state["iteration"] >= state.get("max_iterations", MAX_DEPTH):
        return "done"
    return "generate"  # Generate new image with rewritten prompt


# ═══════════════════════════════════════════════════════════════
# Build LangGraph
# ═══════════════════════════════════════════════════════════════

def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    
    graph.add_node("generate", generate_node)       # Generate/edit image
    graph.add_node("evaluate", evaluate_node)        # Evaluate against criteria
    graph.add_node("aggregate", aggregate_node)      # Score + backpropagate
    graph.add_node("reflect", reflect_and_rewrite_node)  # Reflect + rewrite prompt
    
    graph.set_entry_point("generate")
    
    # generate → evaluate → aggregate → reflect
    graph.add_edge("generate", "evaluate")
    graph.add_edge("evaluate", "aggregate")
    graph.add_edge("aggregate", "reflect")
    
    # reflect → generate again (with new prompt) or done
    graph.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "generate": "generate",
            "done": END,
        },
    )
    
    return graph.compile()


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

async def run_agent(
    prompt: str,
    criteria: list[str],
    on_iteration: callable = None,
    input_image: "Image.Image | None" = None,
) -> dict:
    """
    Run the self-correcting image generation agent.
    
    RESTART MECHANISM:
    - Runs up to MAX_ITERATIONS (5) per attempt
    - If not all criteria pass, RESTARTS fresh generation
    - Episodic memory carries across restarts (learns from past attempts)
    - Max restarts = MAX_RESTARTS (default 2), so total attempts = 3
    - Best image across ALL attempts is preserved
    
    Args:
        prompt: Text description of the image to generate
        criteria: List of criteria the image must satisfy
        on_iteration: Optional async callback for live updates
    
    Returns:
        Full agent response with generated images and evaluation results
    """
    # Shared across restarts
    memory = EpisodicMemory()
    memory.add_prompt(prompt)
    all_history = []
    all_images = []
    global_best_score = 0.0
    global_best_image = None
    global_best_results = []
    global_best_prompt = prompt
    total_iteration_count = 0
    
    for attempt in range(1 + MAX_RESTARTS):
        agent = build_agent_graph()
        root = TreeNode(prompt=prompt)
        
        # On restart: use the best prompt found so far as starting point
        start_prompt = memory.best_prompt if (attempt > 0 and memory.best_prompt) else prompt
        
        # Notify frontend of restart
        if attempt > 0 and on_iteration:
            await on_iteration({
                "iteration": total_iteration_count,
                "overall_score": memory.best_score,
                "all_passed": False,
                "restart": True,
                "attempt": attempt + 1,
                "message": f"Restarting (attempt {attempt + 1}/{1 + MAX_RESTARTS}) — fresh generation with accumulated knowledge",
                "criteria_results": [],
                "prompt_used": start_prompt[:200],
                "image_b64": None,
            })
        
        initial_state: AgentState = {
            "original_prompt": prompt,
            "criteria": criteria,
            "current_prompt": start_prompt,
            "current_image": input_image if attempt == 0 else None,  # Use uploaded image on first attempt
            "iteration": 0,
            "max_iterations": MAX_DEPTH,
            "history": [],
            "memory": memory,  # Carries across restarts!
            "tree_root": root,
            "current_node": root,
            "current_results": [],
            "done": False,
            "images": [],
            "api_calls": 0,
        }
        
        final_state = None
        async for state in agent.astream(initial_state):
            for node_name, node_state in state.items():
                if node_name == "aggregate" and on_iteration:
                    history = node_state.get("history", [])
                    if history:
                        iter_data = history[-1].copy()
                        # Add global context to iteration data
                        iter_data["attempt"] = attempt + 1
                        iter_data["global_iteration"] = total_iteration_count + iter_data.get("iteration", 0)
                        await on_iteration(iter_data)
                if node_state:
                    final_state = node_state
        
        if final_state is None:
            result = await agent.ainvoke(initial_state)
            final_state = result
        
        # Collect results from this attempt
        attempt_history = final_state.get("history", [])
        attempt_images = final_state.get("images", [])
        
        # Tag each record with attempt number
        for h in attempt_history:
            h["attempt"] = attempt + 1
        for img in attempt_images:
            img["attempt"] = attempt + 1
        
        all_history.extend(attempt_history)
        all_images.extend(attempt_images)
        total_iteration_count += len(attempt_history)
        
        # Update global best from memory
        mem: EpisodicMemory = final_state.get("memory", memory)
        memory = mem  # Carry forward
        
        if mem.best_score > global_best_score:
            global_best_score = mem.best_score
            global_best_image = mem.best_image
            global_best_results = mem.best_results
            global_best_prompt = mem.best_prompt
        
        # Check if we're done
        all_passed = all(r.get("passed", False) for r in mem.best_results) if mem.best_results else False
        if all_passed:
            break
        
        # If not done but out of restarts, stop
        if attempt >= MAX_RESTARTS:
            break
        
        # Add restart reflection to memory
        memory.add_reflection({
            "iteration": f"restart_{attempt + 1}",
            "prompt": start_prompt[:200],
            "score": round(mem.best_score, 3),
            "failures": "Exhausted iterations, restarting fresh",
            "fixes": "Try a fundamentally different approach to the prompt",
            "lesson": f"After {len(attempt_history)} iterations, best score was {mem.best_score:.2f}. Need fresh approach.",
        })
    
    # Final results
    all_passed = all(r.get("passed", False) for r in global_best_results) if global_best_results else False
    
    # ── RESOLUTION-AWARE FINAL POLISHING ──────────────────────
    # If best image was generated at low res (early iterations),
    # do one final 1024px pass using the best prompt for high-quality output.
    if global_best_image is not None and global_best_prompt:
        try:
            from app.generator import generate_image
            from app.config import PROGRESSIVE_IMAGE_SIZES
            
            # Regenerate at full resolution with the best prompt
            polished = await generate_image(
                global_best_prompt,
                image_size=PROGRESSIVE_IMAGE_SIZES["final"],
            )
            
            # Quick parallel eval with polish context
            polish_context = memory.get_context() + "\nFinal polish pass at 1024px."
            polish_tasks = [
                evaluate_criterion(
                    polished, c,
                    original_prompt=prompt,
                    context=polish_context,
                )
                for c in criteria
            ]
            polish_results = await asyncio.gather(*polish_tasks)
            polish_score = sum(r["score"] for r in polish_results) / len(polish_results)
            
            # Keep polished version only if at least as good
            if polish_score >= global_best_score - 0.03:
                global_best_image = polished
                global_best_score = max(global_best_score, polish_score)
                global_best_results = list(polish_results)
                all_passed = all(r.get("passed", False) for r in global_best_results)
                
                # Add to images list
                all_images.append({
                    "iteration": total_iteration_count + 1,
                    "prompt": global_best_prompt[:300],
                    "image_b64": image_to_base64(polished),
                    "mode": "polish_1024",
                    "resolution": PROGRESSIVE_IMAGE_SIZES["final"],
                    "attempt": attempt + 1,
                })
                total_iteration_count += 1
                
                if on_iteration:
                    await on_iteration({
                        "iteration": total_iteration_count,
                        "overall_score": round(polish_score, 3),
                        "all_passed": all_passed,
                        "criteria_results": list(polish_results),
                        "prompt_used": global_best_prompt[:200],
                        "image_b64": image_to_base64(polished),
                        "polish": True,
                        "message": f"Final 1024px polish: {polish_score:.2f}",
                    })
        except Exception:
            pass  # Polishing is best-effort, don't fail the whole run
    
    # LATS tree stats
    tree_stats = {
        "total_nodes": total_iteration_count,
        "prompts_tried": memory.prompts_tried,
        "restarts": min(attempt + 1, 1 + MAX_RESTARTS),
    }
    
    if all_passed:
        summary = f"✓ All {len(criteria)} criteria passed after {total_iteration_count} iteration(s). Score: {global_best_score:.2f}"
    else:
        failed = [r["criterion"] for r in global_best_results if not r.get("passed", False)]
        summary = (
            f"Best result after {total_iteration_count} iteration(s) ({attempt + 1} attempt(s)): "
            f"{global_best_score:.2f}. Failed: {', '.join(failed)}"
        )
    
    return {
        "status": "success" if all_passed else "best_effort",
        "total_iterations": total_iteration_count,
        "total_attempts": min(attempt + 1, 1 + MAX_RESTARTS),
        "final_score": round(global_best_score, 3),
        "final_prompt": global_best_prompt,
        "original_prompt": prompt,
        "criteria_results": global_best_results,
        "iteration_history": all_history,
        "mistake_log": [r.get("lesson", "") for r in memory.reflections],
        "tree_search": tree_stats,
        "images": all_images,
        "best_image_b64": image_to_base64(global_best_image) if global_best_image else None,
        "summary": summary,
    }
