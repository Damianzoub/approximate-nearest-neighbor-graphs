# Darth HNSW (Distribution-Aware Repairing and Terminating HNSW)

## 0. Algorithm Explanation (Beginner-Friendly)

### What is the problem?
HNSW uses fixed efSearch for all queries, but:
- Easy queries need less search
- Hard queries need more search
- Same efSearch wastes time on easy queries

X Too much search = slow
X Too little search = low recall

---

### Core intuition (VERY important)

Think of it like a doctor:

- Not every patient needs the same treatment
- Easy queries = quick checkup
- Hard queries = thorough examination

DARTH adapts the search depth based on query difficulty.

---

### Structure

DARTH is built on HNSW with added intelligence:

- Monitors search progress
- Detects when results stabilize
- Decides when to stop early

---

### Two phases

#### 1. Index Construction (same as HNSW)

DARTH uses standard HNSW building:
- Multi-layer graph
- M neighbors per node
- efConstruction parameter

---

#### 2. Search (adaptive termination)

Key insight: Track result improvement over iterations.

For each iteration:
1. Check current best candidates
2. Compare to previous iteration
3. If improvement < threshold -> stop
4. Otherwise -> continue

---

### Key Parameters

- min candidates -> minimum search before checking
- improvement threshold -> when to stop
- window size -> how many iterations to compare

Small threshold = aggressive early stopping
Large threshold = thorough search

---

## 1. Problem the Paper Solves

HNSW wastes computation on easy queries.

DARTH solves:
- Adaptive search termination
- Faster queries without losing recall
- Query difficulty detection

---

## 2. Core Idea

Monitor the search process and stop when results stop improving.

If the best candidates are not changing much, keep searching is unlikely to help.

---

## 3. How It Works (Mechanism)

### Search Phase
1. Start normal HNSW search
2. After min_candidates iterations:
   - Compare current results to previous window
   - Calculate improvement ratio
3. If improvement < threshold:
   - Stop search and return results
4. Otherwise:
   - Continue exploring

Key components:
- Result history buffer
- Improvement calculator
- Early exit decision

---

## 4. Key Formula / Signal

Improvement ratio:
improvement = (prev_best - curr_best) / prev_best

Stop when:
improvement < epsilon

Where epsilon is a small threshold (e.g., 0.01)

---

## 5. Where It Fits (Big Picture)

Category:
- Graph-based ANN with adaptive search

Relation:
- HNSW -> static search (fixed efSearch)
- DARTH -> adaptive stopping based on convergence

Extends HNSW with:
- Query difficulty estimation
- Automatic termination

---

## 6. Strengths

- Saves time on easy queries
- No manual efSearch tuning needed
- Maintains recall on hard queries
- Simple to implement

---

## 7. Weaknesses / Limitations

- Threshold must be tuned
- May stop too early on some hard queries
- Improvement measurement can be noisy
- Only works well when results actually converge

---

## 8. My Implementation Notes

- Built on top of HNSW implementation
- Added:
  - Result history tracking
  - Improvement calculation
  - Adaptive termination logic

Parameters used:
- min_candidates = 10
- improvement_threshold = 0.01
- window_size = 3

---

## 9. Experimental Impact

- Speeds up easy queries by 2-3x
- Maintains recall on hard queries
- Better than fixed efSearch when:
  - Query distribution varies
  - Mix of easy/hard queries

---

## 10. Comparison to Other Methods

- vs HNSW:
  - HNSW = fixed search budget
  - DARTH = adaptive budget

- vs PiP:
  - DARTH = based on result convergence
  - PiP = based on candidate set diversity

- vs AdaEF:
  - DARTH = binary stop/continue decision
  - AdaEF = gradual ef adjustment

---

## 11. Key Insight

Query difficulty varies significantly.

Static search parameters cannot adapt to this variation.

Adaptive termination based on result improvement is a simple but effective solution.

---

## 12. Open Questions / Ideas

- Can we predict difficulty before searching?
- Can we combine with AdaEF for hybrid approach?
- What about queries with multiple clusters?
- How does this work with updates/insertions?

---

## 13. Relationship to Other Papers

This paper inspired:
- PiP (uses similar convergence detection)
- AdaEF (uses similar difficulty estimation)

All three papers try to solve the same problem: when to stop searching.
