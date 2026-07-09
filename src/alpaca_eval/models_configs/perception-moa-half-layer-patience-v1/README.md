Project Perception: Solstice Architecture
The Democratization of Digital Cognition
Project Perception introduces Solstice, a highly specialized, proprietary backend engine. Solstice is an evolutionary step forward in AI architecture—a multi-round, asymmetric "Half-Layer" Mixture-of-Agents (MoA) designed specifically to push the boundaries of multi-model consensus, mitigate hallucinations, and ensure computational efficiency.
The Solstice Deliberation Framework: A Half-Layer Architecture
Traditional MoA frameworks are often symmetric, passing information sequentially without an adversarial check. Solstice breaks this mold by introducing five distinct, highly calculated phases in its Asymmetric Deliberation Loop:
Phase 1: Round 1 Parallel Proposition
To prevent early cognitive homogenization, Solstice isolates its initial reasoning vectors. The prompt is sent to a pool of parallel proposer models, each assigned a distinct persona (e.g., Analyst, Pragmatist, Contextualiser, Creative). They execute asynchronously with zero cross-model communication to generate isolated candidate solutions.
Phase 2: Round 1.5 Interleaved Contrarian Critique
This is the crucial "Half-Layer." Instead of blindly aggregating first-round answers, Solstice passes them to a dedicated Contrarian Reviewer model. This adversarial reviewer cross-examines the parallel solutions to isolate design flaws, unstated assumptions, and technical contradictions, producing a harsh critique brief.
Phase 3: Round 2 Refined Proposition and Convergence
The original specialist models are re-engaged. Provided with their initial answers, peer answers, and the blistering critique, they must intelligently defend their unique architectures or adaptively converge toward a hardened, superior design.
Phase 4: Round 2.5 Pre-Aggregation Audit Brief
Before final synthesis, the Contrarian Reviewer performs one last structural scan across the refined Round 2 updates, generating a localized audit report that flags any lingering logical gaps or edge-case anomalies.
Phase 5: Final Aggregator Arbitration
The supreme synthesis engine reviews the entire historical ledger—original propositions, intermediate critiques, refined updates, and the final audit. It explicitly cross-checks the solutions against the audit report, resolving remaining conflicts and compiling a verified, finalized output.
Context & Memory Management
To prevent the system from forgetting instructions during long conversations, Solstice employs a Rolling Context Compressor. Rather than appending raw chat histories—which causes severe "token bloat" and degrades model attention—this subsystem dynamically compresses histories into an optimized, indexed JSON matrix tracking summaries, facts, preferences, and active topics.
Solstice Versus Traditional MoA
Unlike symmetric frameworks (such as Together MoA) that often fall victim to the "bandwagon effect"—where aggregator models simply agree with flawed premises from first-layer models—Solstice deliberately introduces intellectual friction.
Profound Reduction in Hallucination: The adversarial cross-examination ensures factual inaccuracies are challenged mid-thought, breaking the cycle of confident inaccuracy.
Superior Structural Consistency: Forced defense and refinement lead to remarkably robust and hardened outputs.
Dedicated Safety Filters: A high-priority safety check uses a specialized reviewer to filter malicious requests before executing the heavier core pipeline.
Multimodal Capabilities: Processes file attachments up to 10MB and extracts base64 image data to provide multimodal context.
Production-Ready Reliability: Built for fault tolerance, featuring timeout controllers and automatic retries with exponential backoff.
Model Name: Perception MoA (Half-Layer) - Patience v1
AlpacaEval 2.0 Results (Local)
Length-Controlled Win Rate: 89.11%
Raw Win Rate (Hard): 91.80%
Raw Win Rate (Weighted): 91.58%
Evaluated Examples: 805
- Proposers generate initial responses
- Aggregator iteratively refines the response over multiple "patience steps"
- At each step, the aggregator critiques and improves the current best response
- Early stopping when quality converges (no significant improvement)

This mimics human iterative refinement: draft -> review -> polish -> finalize.

### Model Configuration

| Component | Details |
|-----------|---------|
| **Proposers** | 4-8 diverse LLMs (open/closed) |
| **Aggregator** | Strong instruction-tuned model (e.g., Llama-3-70B, GPT-4) |
| **Patience Steps** | 3-5 iterative refinements |
| **Selection** | Quality-weighted voting + aggregator synthesis |

### AlpacaEval Results (Local)

- **Length-Controlled Win Rate**: 89.11%
- **Raw Win Rate (Hard)**: 91.80%
- **Raw Win Rate (Weighted)**: 91.58%
- **Evaluated Examples**: 805

### Key Innovations

1. **Compute Efficiency**: Half-layer saves ~50% latency vs full MoA
2. **Quality via Iteration**: Patience steps compensate for fewer layers
3. **Diversity Preservation**: Multiple proposers prevent mode collapse
4. **Adaptive Depth**: Early stopping avoids wasted computation

### Reproduction

```bash
# Generate outputs (already done - see results/)
python generate_moa_outputs.py --model perception-moa-half-layer-patience-v1

# Evaluate with AlpacaEval
alpaca_eval evaluate \
  --model_outputs results/perception-moa-half-layer-patience-v1/model_outputs.json \
  --annotators_config weight_scorer
```

### Citation

If you use this architecture, please cite:
```bibtex
@misc{perception-moa-2026,
  title={Perception MoA: Half-Layer Mixture-of-Agents with Patience Aggregation},
  author={Perception AI Team},
  year={2026}
}
```
