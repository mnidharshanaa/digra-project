# DIGRA + RAG + Memory: Mitigating Hallucination Propagation in Multi-Agent Debate

Extends Zhang et al., *"Beware of the Woozle Effect"* (IEEE TASLP, 2026) — reproduces
their DIGRA framework, then adds an evidence-grounded correction (RAG) and a
persistent trust/memory mechanism to address the paper's own documented
"faithfulness hallucination" failure mode (Section III-C1).

## Project status

Modules are built and verified one at a time, in dependency order. A module is
only considered "done" once its unit tests pass — see `tests/`.

| # | Module | Status |
|---|--------|--------|
| 0 | Project scaffold, config, logging, checkpointing | ✅ done |
| 1 | Data layer — FARM loader (schema-verified against live repo) | ✅ done |
| 1b | Response pool builder (Appendix B-3) + LLM client abstraction | ✅ done (logic verified; VLLMClient verified on Kaggle T4x2) |
| 2 | Core debate engine — Standard MAD, fully-connected topology, batched round-2+ generation | ✅ done (orchestration fully tested; VLLMClient wiring untestable here) |
| 3 | Propagation metrics (MA/MR/IMR/CR) + cost instrumentation | ⬜ next |
| 3 | Propagation metrics (MA/MR/IMR/CR) + cost instrumentation | ⬜ |
| 4 | Baselines (CoT, CoT-SC, MAD variants) | ⬜ |
| 5 | DIGRA core (entropy, IG, IGR, partner selection, early stopping) | ⬜ |
| 6 | RAG correction module | ⬜ |
| 7 | Memory / trust module | ⬜ |
| 8 | Experiment orchestrator | ⬜ |
| 9 | Aggregation + figure generation | ⬜ |

## Repository layout

```
configs/            YAML configs. base.yaml is the single source of truth for
                     every dataset path, model id, seed, and hyperparameter.
                     Ablations are override YAMLs merged on top (see
                     src/utils/config.py), never hardcoded in code.
src/
  data/              FARM dataset loading, correct/incorrect response pools
  agents/            Agent / Debate classes, generation wrapper
  entropy/           Mean token entropy, forced decoding
  digra/             IG, IGR, partner selection, early stopping, DIG ablation
  rag/                Retriever, contradiction detector, evidence-adjusted IGR
  memory/            Persistent trust score
  baselines/         CoT, CoT-SC, MAD standard/sparse/random
  metrics/           MA/MR/IMR/CR + timing/token cost instrumentation
  utils/             config, seeding, logging, checkpoint/resume
scripts/             Thin, numbered entry points (01_build_pools.py, etc.)
                     called from Kaggle notebooks — no logic lives here,
                     only orchestration of src/ calls.
notebooks/           Kaggle notebooks; import scripts/src, do not contain logic
tests/               One test file per src/ module; run before any GPU time
                     is spent using that module
results/
  raw/               Per-run CSV rows (one row per dataset/model/method/seed/round)
  aggregated/         Mean±std tables, paper-figure-equivalent data
  figures/           Generated plots, one script per paper-figure-equivalent
```

## Conventions

- **No hardcoded constants in `src/`.** Everything configurable lives in
  `configs/base.yaml`. If you find yourself typing a dataset path, seed, or
  hyperparameter directly into a `.py` file, it belongs in the config instead.
- **Every module ships with tests before it's used in a real experiment.**
  Entropy/IGR/metrics math especially — these are cheap to unit-test and
  expensive to debug after burning GPU hours on a wrong formula.
- **Logging, not print.** `from src.utils.logging_config import get_logger`.
- **Reproducibility.** The same `project.seeds` list (configs/base.yaml) is
  reused identically across every method/dataset/model combination, and
  DIGRA/RAG/Memory variants always start from the same seeded round-1 state
  as Standard MAD (mirrors Appendix B-1 of the DIGRA paper) — this is what
  makes the comparison fair, so never let a method get its own independent
  round-1 sampling.

## Kaggle session safety (read this before starting a long run)

A real Kaggle session was lost mid-pool-building because `/kaggle/working`
was never persisted anywhere outside the live kernel session, and Kaggle's
"Save Version" failed while the GPU cell was still actively running (a
known, common Kaggle platform issue — don't try to Save Version mid-run).
Everything computed in that session (~1 hour of NQ pool-building) was gone
on restart, with no way to recover it.

**Two-part mitigation, both required:**

1. **Clone into a genuinely empty directory.** A doubled path like
   `digra-project/digra-project/` (from running `git clone <url>` a second
   time inside an already-cloned folder) silently causes `results/` to end
   up somewhere other than where you expect, making backups/checks look
   like they're finding nothing even when data exists. Always verify with
   `find /kaggle/working -name results -type d` before assuming a path.

2. **Back up `results/` manually and periodically — don't wait for "Save
   Version" to work.** In a separate notebook cell (safe to run alongside
   a long GPU cell), periodically:
   ```python
   !cd /kaggle/working/digra-project && zip -r /kaggle/working/results_backup.zip results/checkpoints results/pools results/debates
   ```
   Then download `results_backup.zip` via Kaggle's file browser — this
   doesn't require a successful notebook commit. Do this every time you
   step away from a long-running session, not just at the end.

`n_questions` was also cut from 150 to 30 per dataset after this incident,
specifically to bound how much is ever at risk in a single uninterrupted
run, and to preserve GPU budget for the actual DIGRA/RAG/Memory comparison
(Modules 5-7) rather than spending most of it on baseline pool-building.



1. Push this repo to GitHub (private is fine), or upload it as a Kaggle
   Dataset (Kaggle notebooks can attach both a code repo and a dataset).
   Note: `python scripts/00_fetch_farm.py` clones the FARM dataset repo
   directly (needs internet access, which Kaggle notebooks have by default
   unless you're in a no-internet competition environment) — no need to
   bundle the raw data files yourself.
2. **First real GPU run, in order:**
   a. `python scripts/00_fetch_farm.py` — stages the dataset, verifies line counts.
   b. `python scripts/smoke_test_vllm.py --model <hf_id> --tensor-parallel-size <N>` —
      verifies VLLMClient behaves as the LLMClient contract requires. Adjust
      `--dtype`/`--max-model-len`/`--gpu-memory-utilization`/`--tensor-parallel-size`
      to your GPU (T4 needs `--dtype float16`, native bfloat16 isn't supported).
      Read the `[4/4]` output manually before trusting anything built on top of it.
   c. `python scripts/01_build_pools.py` — builds correct/incorrect response
      pools per Appendix B-3.
   d. `python scripts/02_run_debates.py` — runs Standard MAD debates across
      every dataset/model/setup/seed combination, seeded from (c)'s pools.

3. **Scale strategy.** `configs/base.yaml` intentionally runs a reduced
   first pass (`seeds: [0]`, `agent_counts: [3]`, but the full
   `n_questions` per dataset) — enough to validate the entire pipeline
   through Module 3's metrics without committing your full GPU quota
   up front. Once that's confirmed working, restore the full sweep with:
   ```
   python scripts/01_build_pools.py --overrides configs/full_scale.yaml
   python scripts/02_run_debates.py --overrides configs/full_scale.yaml
   ```
   This is always safe to run at any time — checkpoint resume means it
   only computes newly-added seed/agent-count combinations, never redoes
   or invalidates anything already completed.

4. **Performance note.** `run_debate`'s round 2+ loop batches all agents'
   prompts into a single `generate_batch()` call per round (see
   `src/llm/client.py`'s docstring), instead of one call per agent. This
   matters a lot: for a 5-agent debate, that's 5x fewer individual vLLM
   calls per round, and running at batch size >1 uses the GPU far more
   effectively than a Python loop calling `generate()` one agent at a
   time ever could. If you're resuming a run that started before this fix
   (i.e. any `results/` data generated by an earlier commit), it's still
   valid and compatible — only the *speed* changed, not the output format
   — no need to discard or redo anything already completed.
2. In the Kaggle notebook: `!pip install -r requirements.txt`, then
   `import sys; sys.path.append("/kaggle/working/digra-project")`.
3. Point `project.output_root` (configs/base.yaml) at a path under
   `/kaggle/working/` for the session, and periodically save
   `results/` as a Kaggle Dataset version so it survives across sessions —
   the `RunRegistry` (src/utils/checkpoint.py) will pick up exactly where
   it left off on the next session as long as `results/checkpoints/` was
   restored from that saved dataset version.
4. Run `python -m pytest` at the start of every session before spending any
   GPU time, as a sanity check that nothing broke on re-attach.

## Running tests locally

```bash
pip install -r requirements.txt
python -m pytest -v
```
