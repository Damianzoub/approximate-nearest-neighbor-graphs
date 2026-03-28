# AdaEF: Distribution-Aware Exploration for Adaptive HNSW Search

## 0. Algorithm Explanation (Beginner-Friendly)

### What is the problem?
HNSW uses fixed efSearch, but:
- Easy queries need small efSearch (faster)
- Hard queries need large efSearch (better recall)
- Fixed efSearch is a compromise that is never optimal

X Small ef = bad recall on hard queries
X Large ef = slow on easy queries

---

### Core intuition (VERY important)

Think of it like studying for exams:

- Easy topics = quick review (small ef)
- Hard topics = thorough study (large ef)
- One-size-fits-all studying is inefficient

AdaEF adjusts efSearch based on query difficulty estimated during search.

---

### Structure

AdaEF extends HNSW with:

- Query difficulty estimation
- Dynamic efSearch adjustment
- Distribution-aware stopping

---

### Two phases

#### 1. Index Construction (same as HNSW)

AdaEF uses standard HNSW building:
- Multi-layer graph
- M neighbors per node
- efConstruction parameter

---

#### 2. Search (adaptive efSearch)

Key innovation: Estimate difficulty and adjust efSearch dynamically.

During search:
1. Track candidate distribution statistics
2. Estimate how "hard" the query is
3. Adjust efSearch based on difficulty
4. Stop when confident in results

---

### Key Parameters

- Initial efSearch -> starting search budget
- Distribution threshold -> when to increase ef
- Confidence level -> when to stop

Higher thresholds = more aggressive early stopping

---

## 1. Problem the Paper Solves

Fixed efSearch cannot adapt to query difficulty.

AdaEF solves:
- Query difficulty estimation
- Dynamic efSearch adjustment
- Optimal recall-speed trade-off

---

## 2. Core Idea

Estimate query difficulty from the distribution of candidate distances during search.

If candidates are clustered closely together -> easy query
If candidates are spread out -> hard query

---

## 3. How It Works (Mechanism)

### Difficulty Estimation
1. During search, collect candidate distances
2. Compute statistics (variance, spread, etc.)
3. Use these to estimate difficulty

Key signals:
- Distance variance among top candidates
- Gap between k-th and (k+1)-th candidate
- Rate of improvement over iterations

### Adaptive efSearch
1. Start with small initial efSearch
2. After each iteration:
   - Check difficulty estimate
   - If results stable -> reduce efSearch
   - If results unstable -> increase efSearch
3. Stop when confident

### Distribution-Aware Stopping
Stop when:
- Candidates are clustered (easy query)
- Improvement rate is low (no more gains)

---

## 4. Key Formula / Signal

Distance gap ratio:
gap_ratio = (d_{k+1} - d_k) / d_k

If gap_ratio > threshold -> likely found k nearest neighbors

Confidence score:
confidence = 1 - (variance / max_variance)

Higher confidence = easier query = can stop earlier

---

## 5. Where It Fits (Big Picture)

Category:
- Graph-based ANN with adaptive search

Relation:
- HNSW -> static efSearch
- AdaEF -> dynamic efSearch based on difficulty

Extends HNSW with:
- Query difficulty estimation
- Adaptive exploration
- Distribution analysis

---

## 6. Strengths

- Optimal recall-speed trade-off per query
- No manual efSearch tuning needed
- Works well on mixed query distributions
- Principled approach (based on distribution analysis)

---

## 7. Weaknesses / Limitations

- Complexity in difficulty estimation
- Requires careful threshold tuning
- May be sensitive to data distribution
- Computational overhead for statistics

---

## 8. My Implementation Notes

- Built on top of HNSW implementation
- Added:
  - Distance statistics tracker
  - Difficulty estimator
  - Dynamic efSearch controller

Parameters used:
- initial_ef = 16
- min_ef = 8
- max_ef = 200
- gap_threshold = 0.1

---

## 9. Experimental Impact

- Similar recall to high efSearch
- 2-5x speedup on easy queries
- Minimal overhead for difficulty estimation
- Best for:
  - Mixed query distributions
  - Unknown query difficulty a priori

---

## 10. Comparison to Other Methods

- vs HNSW:
  - HNSW = fixed efSearch
  - AdaEF = adaptive efSearch

- vs DARTH:
  - DARTH = binary stop/continue
  - AdaEF = gradual ef adjustment

- vs PiP:
  - PiP = parallel exploration
  - AdaEF = sequential adaptive exploration

---

## 11. Key Insight

Query difficulty can be estimated from candidate distribution during search.

This allows efSearch to adapt dynamically, achieving optimal recall-speed trade-off for each query.

---

## 12. Open Questions / Ideas

- Can we predict difficulty before search starts?
- Can we use ML to learn difficulty thresholds?
- How to handle queries with multiple clusters?
- Can AdaEF combine with PiP for parallel adaptive search?

---

## 13. Relationship to Other Papers

Addresses the same problem as:
- DARTH (when to stop)
- PiP (how to explore efficiently)

Unique contribution:
- Distribution-aware difficulty estimation
- Gradual efSearch adjustment
- Principled stopping criteria

Future work:
- Combine with PiP for parallel adaptive search
- Learn thresholds from data
- Handle streaming queries
