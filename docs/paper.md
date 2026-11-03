# Paper reference and implementation scope

[Back to README](../README.md) · [Architecture](architecture.md) · [Run guide](reproduction.md)

## Reference version

Duo Xu and Faramarz Fekri, **NeSyFS: A Neuro-symbolic Fast-Slow Thinking Framework for LLM Agent under Partial Observability**, arXiv:2607.28942v2, 3 August 2026.

[Versioned abstract](https://arxiv.org/abs/2607.28942v2) · [Versioned HTML](https://arxiv.org/html/2607.28942v2) · [Bundled PDF](../2607.28942v2.pdf) · [BibTeX](reference.bib)

The bundled PDF is the reference for this documentation. Page numbers below are PDF page numbers. This repository is an independent implementation inspired by the method; the paper's authorship does not imply authorship or endorsement of this code.

## Method in brief

The paper uses a knowledge graph as structured memory under partial observability. A fast policy conditions on retrieved triplets. A stepwise reflection module checks candidate actions, with repeated rejection triggering a slower TSMC-style planner. Planning predicts future observations and graph states, assigns binary-progress-based weights, resamples candidate trajectories and executes one selected action. See Methodology on pages 3 to 5 plus Algorithms 1 and 2 on page 10.

The paper itself describes a TSMC-style method without full importance-weight computation. Do not treat this implementation as an exact posterior sampler or a formal guarantee of successful planning.

## Paper-to-code map

| Aspect | Paper v2 | This repository |
| --- | --- | --- |
| Memory update | Extract triplets, filter already-present triplets, remove outdated knowledge (page 11). | `MemoryDelta` inserts facts with new UUIDs and removes explicitly superseded IDs. No semantic deduplication. |
| Retrieval | Retrieve relevant incident triplets up to a predefined graph depth (page 11). | Exact entity-name matching with incident-fact lookup and a default cap of 100. No multi-hop traversal. |
| Fast policy | LLM action proposal using task, observation and retrieved triplets (page 3). | LangChain `create_agent` with structured `ActionProposal`; execution remains outside the policy. |
| Reflection | Five votes, 80% approval; switch after repeated rejection with `K=3` (pages 4 and 10). | Five votes, `ceil(0.8 * samples)` approval and up to three proposals by default. |
| Simulated state | Initialize particles by copying the current KG (pages 5 and 10). | Start from the retrieved fact list, not a complete Neo4j graph copy. Simulated updates stay in memory. |
| Progress judgment | Predict observation, update simulated KG, then judge progress using new triplets (pages 5 and 10). | Predict observation and progress together, then extract the simulated memory delta. |
| Particle settings | `N=6`, `L=5`, progress-weight parameter `lambda=1` (page 10). | Six particles, five future rollout rounds, weight multiplier `exp(1)` for progress. Algorithm 2 counts the initial state within `L`, so the rollout-length conventions differ. |
| Resampling and selection | Weighted particle resampling followed by uniform final selection (pages 5 and 10). | Weighted resampling with replacement; reset weights after resampling; return the first action of a uniformly selected survivor. |
| Models and sampling | Evaluations use GPT-5, GPT-5-mini and Llama-3.3-70B-Instruct (page 5). | Default `gpt-4.1-mini`; all calls use generation temperature `0`. Multiple votes/particles do not guarantee diverse samples. |
| Evaluation | ALFWorld, WebShop and ScienceWorld; reported results averaged across three seeds (page 5). | Small kitchen simulator and manual mode. No benchmark integrations or evaluation harness. |

The matching parameter values are useful starting points. They do not establish a faithful reproduction of the paper's experiments.

## Paper-reported results

The following **success rates (%) are transcribed from Table 1 on PDF page 6**. They are the authors' reported NeSyFS results, not measurements from this repository.

| Base model in the paper | ALFWorld SR | WebShop SR | ScienceWorld SR |
| --- | ---: | ---: | ---: |
| GPT-5-mini | 91.1 | 51.3 | 61.2 |
| GPT-5 | 93.6 | 63.3 | 69.4 |
| Llama-3.3-70B | 89.6 | 62.5 | 67.1 |

SR denotes success rate. WebShop and ScienceWorld also have separate average-reward columns in the source table; those are not mixed into this table. The source's aggregate “Average” column is omitted.

Table 1 compares against ReAct, Reflexion, ABBEL, RAFA and SwiftSage. Consult the paper for the full baselines and evaluation details. The separate controlled planning experiment in Appendix D, Table 4, reports higher success rates for its tree-search baseline at higher token cost, so the paper should not be summarized as universal superiority on every metric.

## What has not been established here

- The paper's benchmark scores have not been reproduced with this repository.
- The current model, prompts, retrieval and rollout conventions differ from the paper.
- Passing offline tests establishes behavior under test fixtures, not the quality of model judgments.
- The graph does not guarantee complete, consistent or probabilistically calibrated beliefs.
- Repeated reflection is not a substitute for application-level safety and authorization checks.

See [reproduction guidance](reproduction.md#benchmark-reproduction-boundary) before designing an evaluation. Cite the paper when discussing the research method; identify this repository separately when reporting software-specific results.
