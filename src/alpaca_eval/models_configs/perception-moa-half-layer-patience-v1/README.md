# Perception MoA (Half-Layer) - Patience v1

## Architecture Overview

**Perception MoA** is a Mixture-of-Agents (MoA) architecture that uses a **half-layer** design with a **Patience** aggregation strategy.

### Half-Layer MoA Design

Traditional MoA uses multiple full layers of agents where each layer's outputs feed into the next. Our half-layer approach:

1. **Single Proposer Layer**: Multiple diverse proposer models generate candidate responses in parallel
2. **Single Aggregator Layer**: One aggregator model synthesizes the best response from all proposals

This reduces latency and compute while maintaining the diversity benefits of MoA.

### Patience Aggregation Strategy

The "Patience" variant implements a **sequential refinement** approach:

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
@misc{perception-moa-2024,
  title={Perception MoA: Half-Layer Mixture-of-Agents with Patience Aggregation},
  author={Perception AI Team},
  year={2024}
}
```
